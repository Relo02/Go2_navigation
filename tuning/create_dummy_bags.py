#!/usr/bin/env python3
"""
Creates synthetic ROS2 bag files for testing bo_gp_analysis.ipynb
without running Gazebo or the real planner.

Generates two scenarios:
  default_config/   — trajectory with default planner_params.yaml weights
  optimized_config/ — trajectory that better avoids obstacles

Usage:
  source /opt/ros/jazzy/setup.bash
  python3 create_dummy_bags.py [output_dir]

Output dir defaults to ../tuning_results/dummy_bags/
"""

import sys
import sqlite3
import time
import math
import numpy as np
from pathlib import Path

# ─── ROS imports (needs sourced ROS env) ──────────────────────────────────────
try:
    import rclpy
    from rclpy.serialization import serialize_message
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Path as NavPath, Odometry
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import PointCloud2, PointField
    import struct
except ImportError as e:
    print(f"ERROR: {e}")
    print("Source ROS first:  source /opt/ros/jazzy/setup.bash")
    sys.exit(1)


# ─── Obstacle layout (world frame) ────────────────────────────────────────────
OBSTACLES = [
    (2.0, 0.5),
    (2.0, -0.5),
    (3.5, 1.2),
    (3.5, -1.2),
    (5.0, 0.8),
]
OBSTACLE_RADIUS = 0.4  # metres

GOAL = (6.0, 0.0)


# ─── CDR bag helpers ──────────────────────────────────────────────────────────

def create_bag_db(path: Path) -> sqlite3.Connection:
    path.mkdir(parents=True, exist_ok=True)
    db = path / "bag_0.db3"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            serialization_format TEXT NOT NULL DEFAULT 'cdr',
            offered_qos_profiles TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            data BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS messages_ts ON messages (timestamp ASC);
    """)
    return conn


def add_topic(conn: sqlite3.Connection, tid: int, name: str, msgtype: str):
    conn.execute(
        "INSERT OR IGNORE INTO topics (id,name,type,serialization_format) VALUES (?,?,?,'cdr')",
        (tid, name, msgtype),
    )


def write_msg(conn: sqlite3.Connection, tid: int, ts_ns: int, msg):
    raw = serialize_message(msg)
    conn.execute(
        "INSERT INTO messages (topic_id, timestamp, data) VALUES (?,?,?)",
        (tid, ts_ns, raw),
    )


def write_metadata_yaml(bag_dir: Path, topics: dict, duration_ns: int, start_ns: int, n_msgs: int):
    topic_lines = ""
    for name, msgtype in topics.items():
        topic_lines += f"""    - topic_metadata:
        name: {name}
        type: {msgtype}
        serialization_format: cdr
        offered_qos_profiles: ''
      message_count: {n_msgs}
"""
    yaml = f"""rosbag2_bagfile_information:
  version: 6
  storage_identifier: sqlite3
  relative_file_paths:
    - bag_0.db3
  duration:
    nanoseconds: {duration_ns}
  starting_time:
    nanoseconds_since_epoch: {start_ns}
  message_count: {n_msgs}
  topics_with_message_count:
{topic_lines}  compression_format: ''
  compression_mode: ''
"""
    (bag_dir / "metadata.yaml").write_text(yaml)


# ─── Message builders ─────────────────────────────────────────────────────────

def make_pose(sec: int, nsec: int, x: float, y: float, yaw: float = 0.0,
              frame: str = "odom") -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = frame
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.orientation.w = math.cos(yaw / 2)
    return msg


def make_twist(vx: float, omega: float) -> Twist:
    msg = Twist()
    msg.linear.x = vx
    msg.angular.z = omega
    return msg


def make_path(sec: int, nsec: int, waypoints: list) -> NavPath:
    path = NavPath()
    path.header.stamp.sec = sec
    path.header.stamp.nanosec = nsec
    path.header.frame_id = "map"
    for x, y in waypoints:
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
    return path


def make_diagnostics(success: float, cost: float, solve_ms: float,
                     avg_ms: float, fails: float, security: float,
                     vx_eff: float) -> Float64MultiArray:
    msg = Float64MultiArray()
    msg.data = [success, cost, solve_ms, avg_ms, fails, security, vx_eff]
    return msg


def make_pointcloud2(sec: int, nsec: int, points_xy: np.ndarray) -> PointCloud2:
    """Create a XYZI float32 PointCloud2 from Nx2 xy array."""
    msg = PointCloud2()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = "map"

    msg.height = 1
    msg.width = len(points_xy)

    fields = []
    for name, offset, dtype in [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)]:
        f = PointField()
        f.name = name
        f.offset = offset
        f.datatype = dtype  # FLOAT32 = 7
        f.count = 1
        fields.append(f)
    msg.fields = fields

    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(points_xy)
    msg.is_dense = True

    buf = bytearray()
    for x, y in points_xy:
        buf += struct.pack("<ffff", float(x), float(y), 0.0, 1.0)
    msg.data = bytes(buf)
    return msg


# ─── Trajectory simulation ────────────────────────────────────────────────────

def simulate_trajectory(
    goal: tuple,
    obstacles: list,
    obs_radius: float,
    avoidance_gain: float,    # higher → pushes robot further from obstacles
    tracking_gain: float,     # higher → follows straight line more aggressively
    dt: float = 0.1,
    max_steps: int = 600,
    vmax: float = 0.8,
):
    """
    Simple 2-D potential-field simulation to produce two qualitatively
    different trajectories (default vs. GP-optimised).
    """
    pos = np.array([0.0, 0.0])
    traj = [pos.copy()]

    for _ in range(max_steps):
        # Attractive force toward goal
        diff_g = np.array(goal) - pos
        dist_g = np.linalg.norm(diff_g)
        if dist_g < 0.15:
            break
        f_att = tracking_gain * diff_g / (dist_g + 1e-6)

        # Repulsive force from obstacles
        f_rep = np.zeros(2)
        for ox, oy in obstacles:
            diff_o = pos - np.array([ox, oy])
            dist_o = np.linalg.norm(diff_o)
            influence = obs_radius * 2.5
            if dist_o < influence:
                scale = avoidance_gain * (1.0 / (dist_o + 0.05) - 1.0 / influence)
                f_rep += scale * diff_o / (dist_o + 1e-6)

        vel = f_att + f_rep
        speed = np.linalg.norm(vel)
        if speed > vmax:
            vel = vel / speed * vmax

        pos = pos + vel * dt
        traj.append(pos.copy())

    return np.array(traj)


def generate_lidar_points(robot_pos: np.ndarray, obstacles: list,
                          obs_radius: float, n_rays: int = 180,
                          max_range: float = 5.0) -> np.ndarray:
    """Simulate a 2-D lidar scan in world frame."""
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    pts = []
    for a in angles:
        direction = np.array([math.cos(a), math.sin(a)])
        hit_dist = max_range
        for ox, oy in obstacles:
            obs = np.array([ox, oy]) - robot_pos
            proj = np.dot(obs, direction)
            if proj <= 0:
                continue
            perp_sq = np.dot(obs, obs) - proj**2
            if perp_sq > obs_radius**2:
                continue
            dist = proj - math.sqrt(max(0, obs_radius**2 - perp_sq))
            if 0 < dist < hit_dist:
                hit_dist = dist
        if hit_dist < max_range:
            hit = robot_pos + direction * hit_dist
            pts.append(hit)
    return np.array(pts) if pts else np.zeros((0, 2))


# ─── Main bag writer ──────────────────────────────────────────────────────────

TOPICS_MAP = {
    "/go2/pose":           "geometry_msgs/msg/PoseStamped",
    "/cmd_vel":            "geometry_msgs/msg/Twist",
    "/a_star/path":        "nav_msgs/msg/Path",
    "/mpc/diagnostics":    "std_msgs/msg/Float64MultiArray",
    "/mpc/next_setpoint":  "geometry_msgs/msg/PoseStamped",
    "/lidar/points_filtered": "sensor_msgs/msg/PointCloud2",
    "/goal_pose":          "geometry_msgs/msg/PoseStamped",
}

TOPIC_IDS = {name: i + 1 for i, name in enumerate(TOPICS_MAP)}


def write_bag(bag_dir: Path, traj: np.ndarray, label: str, dt: float = 0.1):
    bag_dir.mkdir(parents=True, exist_ok=True)
    conn = create_bag_db(bag_dir)

    for name, msgtype in TOPICS_MAP.items():
        add_topic(conn, TOPIC_IDS[name], name, msgtype)

    BASE_NS = 1_776_900_000_000_000_000  # synthetic epoch near 2026
    n_steps = len(traj)
    duration_ns = int(n_steps * dt * 1e9)

    # Goal pose (written once)
    goal_msg = make_pose(0, 0, GOAL[0], GOAL[1], frame="map")
    write_msg(conn, TOPIC_IDS["/goal_pose"], BASE_NS, goal_msg)

    # A* reference path (straight line to goal)
    astar_wps = [(float(i) / 10 * GOAL[0], float(i) / 10 * GOAL[1]) for i in range(1, 11)]

    total_msgs = 1
    for i, pos in enumerate(traj):
        ts_ns = BASE_NS + int(i * dt * 1e9)
        sec = int(ts_ns // 1_000_000_000)
        nsec = int(ts_ns % 1_000_000_000)

        # /go2/pose
        yaw = math.atan2(traj[min(i + 1, n_steps - 1), 1] - pos[1],
                         traj[min(i + 1, n_steps - 1), 0] - pos[0])
        write_msg(conn, TOPIC_IDS["/go2/pose"], ts_ns,
                  make_pose(sec, nsec, pos[0], pos[1], yaw))

        # /cmd_vel
        if i + 1 < n_steps:
            dp = traj[i + 1] - pos
            vx = math.hypot(dp[0], dp[1]) / dt
            dyaw = math.atan2(dp[1], dp[0]) - yaw
            omega = math.atan2(math.sin(dyaw), math.cos(dyaw)) / dt
        else:
            vx, omega = 0.0, 0.0
        write_msg(conn, TOPIC_IDS["/cmd_vel"], ts_ns, make_twist(vx, omega))

        # /a_star/path  (every 10 steps)
        if i % 10 == 0:
            remaining = [(x, y) for (x, y) in astar_wps if x > pos[0] - 0.5][:8]
            if remaining:
                write_msg(conn, TOPIC_IDS["/a_star/path"], ts_ns,
                          make_path(sec, nsec, remaining))

        # /mpc/next_setpoint
        sp_idx = min(i + 5, n_steps - 1)
        write_msg(conn, TOPIC_IDS["/mpc/next_setpoint"], ts_ns,
                  make_pose(sec, nsec, traj[sp_idx, 0], traj[sp_idx, 1], 0.0))

        # /mpc/diagnostics
        solve_ms = 35.0 + np.random.normal(0, 5)
        cost = 5000 + np.random.normal(0, 500)
        diag = make_diagnostics(1.0, cost, solve_ms, solve_ms * 0.9, 0.0, 0.0, 1.0)
        write_msg(conn, TOPIC_IDS["/mpc/diagnostics"], ts_ns, diag)

        # /lidar/points_filtered (every 5 steps)
        if i % 5 == 0:
            pts = generate_lidar_points(pos, OBSTACLES, OBSTACLE_RADIUS)
            if len(pts) > 0:
                write_msg(conn, TOPIC_IDS["/lidar/points_filtered"], ts_ns,
                          make_pointcloud2(sec, nsec, pts))

        total_msgs += 1

    conn.commit()
    conn.close()
    write_metadata_yaml(bag_dir, TOPICS_MAP, duration_ns, BASE_NS, total_msgs)
    print(f"  [{label}] {n_steps} steps → {bag_dir}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).parent.parent / "tuning_results" / "dummy_bags"

    np.random.seed(42)

    configs = {
        "default_config": dict(
            avoidance_gain=0.5,    # weak obstacle repulsion (default)
            tracking_gain=1.5,
        ),
        "optimized_config": dict(
            avoidance_gain=2.5,    # strong obstacle repulsion (GP-optimised)
            tracking_gain=1.2,
        ),
    }

    print(f"Writing dummy bags to: {out_root}")
    for name, kwargs in configs.items():
        traj = simulate_trajectory(GOAL, OBSTACLES, OBSTACLE_RADIUS, **kwargs)
        bag_dir = out_root / name / "bag"
        write_bag(bag_dir, traj, name)

    print("Done. Use these paths in bo_gp_analysis.ipynb:")
    print(f"  DEFAULT_BAG  = '{out_root / 'default_config' / 'bag'}'")
    print(f"  OPTIMIZED_BAG = '{out_root / 'optimized_config' / 'bag'}'")


if __name__ == "__main__":
    main()
