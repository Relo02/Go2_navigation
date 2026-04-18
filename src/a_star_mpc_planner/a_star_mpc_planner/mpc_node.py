"""
MPC tracker ROS2 node for the Go2 quadruped robot.

Architecture
------------
  Subscribes:
    /go2/pose    PoseStamped      — robot pose (position + orientation)
    /lidar/points_filtered PointCloud2     — 3-D lidar hits (world frame)
    /a_star/path nav_msgs/Path    — local A* path

  Publishes:
    /mpc/predicted_path  nav_msgs/Path          — N-step MPC predicted trajectory
    /mpc/next_setpoint   geometry_msgs/PoseStamped — lookahead setpoint
    /mpc/diagnostics     std_msgs/Float64MultiArray — [success, cost, solve_time_ms]

  Control flow (at mpc_rate_hz):
    1. Estimate velocity by differentiating consecutive /go2/pose messages.
    2. Solve MPCTracker → predicted state trajectory.
    3. Walk the predicted trajectory to find the first state >= mpc_lookahead_dist
       from the robot and publish that as the setpoint.
    4. Publish diagnostics.

author: Lorenzo Ortolani (adapted for Go2)
"""

import math
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray

from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap
from a_star_mpc_planner.mpc_tracker import MPCConfig, MPCTracker


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw from quaternion."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _yaw_to_quat(yaw: float) -> tuple:
    """Convert yaw angle to quaternion (roll=0, pitch=0, yaw=yaw)."""
    half_yaw = yaw / 2.0
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    return (0.0, 0.0, sy, cy)  # (qx, qy, qz, qw)


class MPCNode(Node):

    def __init__(self):
        super().__init__('mpc_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('mpc_N',            30)
        self.declare_parameter('mpc_dt',           0.1)
        self.declare_parameter('mpc_vx_max',       1.0)
        self.declare_parameter('mpc_vy_max',       0.5)
        self.declare_parameter('mpc_omega_max',    1.5)
        self.declare_parameter('mpc_v_ref',        0.5)
        self.declare_parameter('mpc_Q_xy',        20.0)
        self.declare_parameter('mpc_Q_yaw',        0.5)
        self.declare_parameter('mpc_Q_terminal',  50.0)
        self.declare_parameter('mpc_R_vel',        1.0)
        self.declare_parameter('mpc_R_omega',      0.5)
        self.declare_parameter('mpc_R_jerk',       0.2)
        self.declare_parameter('mpc_W_obs_sigmoid',       500.0)
        self.declare_parameter('mpc_obs_alpha',           8.0)
        self.declare_parameter('mpc_obs_r',               0.5)
        self.declare_parameter('mpc_max_obs_constraints', 15)
        self.declare_parameter('mpc_obs_check_radius',    2.0)
        self.declare_parameter('mpc_max_iter',   100)
        self.declare_parameter('mpc_warm_start',  True)
        self.declare_parameter('mpc_rate_hz',       2.0)
        self.declare_parameter('mpc_lookahead_dist', 0.5)
        self.declare_parameter('max_lidar_range',  6.0)
        self.declare_parameter('mpc_path_resample_ds', 0.20)
        self.declare_parameter('mpc_path_smooth_window', 5)
        self.declare_parameter('mpc_setpoint_alpha', 0.35)
        self.declare_parameter('mpc_setpoint_max_step', 0.30)
        self.declare_parameter('mpc_setpoint_reset_dist', 1.25)

        # Security protocol parameters
        self.declare_parameter('grid_reso',                  0.25)
        self.declare_parameter('grid_half_width',            5.0)
        self.declare_parameter('grid_std',                   0.2)
        self.declare_parameter('mpc_security_threshold',     0.35)
        self.declare_parameter('mpc_security_escape_radius', 3.0)

        # ── MPCConfig ─────────────────────────────────────────────────
        cfg = MPCConfig(
            N             = int(self.get_parameter('mpc_N').value),
            dt            = float(self.get_parameter('mpc_dt').value),
            vx_max        = float(self.get_parameter('mpc_vx_max').value),
            vy_max        = float(self.get_parameter('mpc_vy_max').value),
            omega_max     = float(self.get_parameter('mpc_omega_max').value),
            v_ref         = float(self.get_parameter('mpc_v_ref').value),
            Q_xy          = float(self.get_parameter('mpc_Q_xy').value),
            Q_yaw         = float(self.get_parameter('mpc_Q_yaw').value),
            Q_terminal    = float(self.get_parameter('mpc_Q_terminal').value),
            R_vel         = float(self.get_parameter('mpc_R_vel').value),
            R_omega       = float(self.get_parameter('mpc_R_omega').value),
            R_jerk        = float(self.get_parameter('mpc_R_jerk').value),
            W_obs_sigmoid        = float(self.get_parameter('mpc_W_obs_sigmoid').value),
            obs_alpha            = float(self.get_parameter('mpc_obs_alpha').value),
            obs_r                = float(self.get_parameter('mpc_obs_r').value),
            max_obs_constraints  = int(self.get_parameter('mpc_max_obs_constraints').value),
            obs_check_radius     = float(self.get_parameter('mpc_obs_check_radius').value),
            max_iter      = int(self.get_parameter('mpc_max_iter').value),
            warm_start    = bool(self.get_parameter('mpc_warm_start').value),
        )
        self._tracker = MPCTracker(config=cfg)

        # ── Security protocol ─────────────────────────────────────────
        self._security_threshold = float(self.get_parameter('mpc_security_threshold').value)
        self._security_escape_radius = float(self.get_parameter('mpc_security_escape_radius').value)
        self._grid = FixedGaussianGridMap(
            reso=float(self.get_parameter('grid_reso').value),
            half_width=float(self.get_parameter('grid_half_width').value),
            std=float(self.get_parameter('grid_std').value),
        )
        self._security_mode: bool = False

        self._max_lidar_range = float(self.get_parameter('max_lidar_range').value)
        self._lookahead_dist  = float(self.get_parameter('mpc_lookahead_dist').value)
        self._path_resample_ds = float(self.get_parameter('mpc_path_resample_ds').value)
        self._path_smooth_window = int(self.get_parameter('mpc_path_smooth_window').value)
        self._setpoint_alpha = float(self.get_parameter('mpc_setpoint_alpha').value)
        self._setpoint_max_step = float(self.get_parameter('mpc_setpoint_max_step').value)
        self._setpoint_reset_dist = float(self.get_parameter('mpc_setpoint_reset_dist').value)

        # ── State ─────────────────────────────────────────────────────
        self._pose: PoseStamped | None = None
        self._yaw = 0.0
        self._a_star_path: list | None = None
        self._a_star_path_raw_len = 0
        self._lidar_points: np.ndarray | None = None
        self._setpoint_filtered_xy: np.ndarray | None = None
        self._setpoint_filtered_yaw: float | None = None

        # Logging counters
        self._solve_count    = 0
        self._fail_count     = 0
        self._total_solve_ms = 0.0

        # ── QoS ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(PoseStamped, '/go2/pose',        self._pose_cb,  10)
        self.create_subscription(PointCloud2, '/lidar/points_filtered',    self._lidar_cb, sensor_qos)
        self.create_subscription(Path,        '/a_star/path',     self._path_cb,  10)

        # ── Publishers ────────────────────────────────────────────────
        self._pred_path_pub = self.create_publisher(Path,                '/mpc/predicted_path', 10)
        self._setpoint_pub  = self.create_publisher(PoseStamped,         '/mpc/next_setpoint',  10)
        self._diagnostics_pub = self.create_publisher(Float64MultiArray, '/mpc/diagnostics',    10)

        # ── Replan timer ──────────────────────────────────────────────
        rate = float(self.get_parameter('mpc_rate_hz').value)
        self.create_timer(1.0 / rate, self._solve_cb)

        self.get_logger().info('MPC node ready')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        """Process pose and extract yaw."""
        self._pose = msg
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self._yaw = _quat_to_yaw(qx, qy, qz, qw)

        # DEBUG: log every received pose update (throttled)
        self.get_logger().info(
            f'[MPC-DEBUG] /go2/pose received: '
            f'pos=({msg.pose.position.x:.4f}, {msg.pose.position.y:.4f})  '
            f'yaw={math.degrees(self._yaw):.1f} deg',
            throttle_duration_sec=2.0,
        )

    def _lidar_cb(self, msg: PointCloud2):
        """Parse LiDAR points and filter by range from robot position."""
        try:
            points = list(point_cloud2.read_points(msg, skip_nans=True))
            if points:
                arr = np.array([(p[0], p[1], p[2]) for p in points], dtype=float)
                if self._pose is not None:
                    px = self._pose.pose.position.x
                    py = self._pose.pose.position.y
                    dists = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
                    arr = arr[dists < self._max_lidar_range]
                self._lidar_points = arr if len(arr) > 0 else None
            else:
                self._lidar_points = None
        except Exception as e:
            self.get_logger().warn(f'Lidar error: {e}')

    def _path_cb(self, msg: Path):
        """
        Store the latest A* path.

        We keep the setpoint filter state across normal replans to avoid jitter.
        Reset happens only when the new path jumps too far from current filtered
        setpoint (e.g. new goal / large topological change).
        """
        if msg.poses:
            raw_path = [
                (p.pose.position.x, p.pose.position.y, p.pose.position.z)
                for p in msg.poses
            ]
            self._a_star_path_raw_len = len(raw_path)
            smoothed_path = self._smooth_resample_path(raw_path)
            self._a_star_path = smoothed_path

            # Keep filter continuity unless the new path is a large jump.
            did_reset = False
            if self._setpoint_filtered_xy is None or self._setpoint_filtered_yaw is None:
                did_reset = True
            else:
                idx = 1 if len(smoothed_path) > 1 else 0
                path_anchor_xy = np.array([
                    float(smoothed_path[idx][0]),
                    float(smoothed_path[idx][1]),
                ], dtype=float)
                jump = float(np.linalg.norm(path_anchor_xy - self._setpoint_filtered_xy))
                if jump > self._setpoint_reset_dist:
                    did_reset = True

            if did_reset:
                self._setpoint_filtered_xy = None
                self._setpoint_filtered_yaw = None

            self.get_logger().info(
                f'[MPC] Received NEW A* path: {len(raw_path)} raw points → '
                f'{len(self._a_star_path)} resampled points '
                f'| reset_filter={did_reset}',
                throttle_duration_sec=1.0,
            )
        else:
            self._a_star_path = None
            self._a_star_path_raw_len = 0
            self._setpoint_filtered_xy = None
            self._setpoint_filtered_yaw = None

    def _smooth_resample_path(self, path_xyz: list) -> list:
        """
        Resample and smooth the A* path before feeding it to MPC.
        This reduces cell-to-cell zig-zag jitter while preserving endpoints.
        """
        if not path_xyz or len(path_xyz) < 2:
            return path_xyz

        xy = np.array([(p[0], p[1]) for p in path_xyz], dtype=float)

        # Remove repeated consecutive points.
        if len(xy) >= 2:
            dxy = np.diff(xy, axis=0)
            keep = np.hstack(([True], np.linalg.norm(dxy, axis=1) > 1e-4))
            xy = xy[keep]
        if len(xy) < 2:
            z = float(path_xyz[-1][2])
            return [(float(xy[0, 0]), float(xy[0, 1]), z)]

        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(arc[-1])
        if total <= 1e-6:
            z = float(path_xyz[-1][2])
            return [(float(xy[0, 0]), float(xy[0, 1]), z), (float(xy[-1, 0]), float(xy[-1, 1]), z)]

        ds = max(self._path_resample_ds, 1e-2)
        s = np.arange(0.0, total + 1e-9, ds)
        if s[-1] < total:
            s = np.append(s, total)

        x = np.interp(s, arc, xy[:, 0])
        y = np.interp(s, arc, xy[:, 1])
        pts = np.column_stack((x, y))

        # Lightweight moving-average smoothing (odd window), keep endpoints fixed.
        win = self._path_smooth_window
        if win >= 3 and (win % 2 == 1) and len(pts) >= win:
            kernel = np.ones(win, dtype=float) / float(win)
            xs = np.convolve(pts[:, 0], kernel, mode='same')
            ys = np.convolve(pts[:, 1], kernel, mode='same')
            pts[:, 0] = xs
            pts[:, 1] = ys
            pts[0] = xy[0]
            pts[-1] = xy[-1]

        z = float(path_xyz[-1][2])
        return [(float(p[0]), float(p[1]), z) for p in pts]

    def _find_escape_target(
        self,
        grid: FixedGaussianGridMap,
        robot_xy: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        BFS over the occupancy grid starting from the robot's cell to find the
        nearest cell whose occupancy probability is below security_threshold.
        Returns the world-frame (x, y) of that cell, or None if not found within
        security_escape_radius.
        """
        ix0, iy0 = grid.world_to_index(float(robot_xy[0]), float(robot_xy[1]))
        if ix0 is None:
            return None

        visited: set = set()
        queue: deque = deque()
        queue.append((ix0, iy0))
        visited.add((ix0, iy0))

        while queue:
            ix, iy = queue.popleft()
            if float(grid.gmap[ix, iy]) < self._security_threshold:
                wx, wy = grid.index_to_world(ix, iy)
                return np.array([wx, wy], dtype=float)
            for dix in (-1, 0, 1):
                for diy in (-1, 0, 1):
                    if dix == 0 and diy == 0:
                        continue
                    nix, niy = ix + dix, iy + diy
                    if (nix, niy) in visited:
                        continue
                    if not (0 <= nix < grid.cells and 0 <= niy < grid.cells):
                        continue
                    wx, wy = grid.index_to_world(nix, niy)
                    if np.hypot(wx - robot_xy[0], wy - robot_xy[1]) > self._security_escape_radius:
                        continue
                    visited.add((nix, niy))
                    queue.append((nix, niy))

        return None

    def _solve_cb(self):
        """Solve MPC and publish results."""
        if self._pose is None or self._a_star_path is None:
            self.get_logger().warn('[MPC] Waiting for pose and path…', throttle_duration_sec=5.0)
            return

        # State: [px, py, yaw] for 2-D kinematic model
        state = np.array([
            self._pose.pose.position.x,
            self._pose.pose.position.y,
            self._yaw,
        ])

        # LiDAR obstacles
        obs_2d = None
        if self._lidar_points is not None and len(self._lidar_points) > 0:
            obs_2d = self._lidar_points[:, :2]

        # ── Security protocol: detect inflated-obstacle collision ────
        in_inflated = False
        escape_target: Optional[np.ndarray] = None
        occ_at_robot = 0.0
        if self._lidar_points is not None and len(self._lidar_points) > 0:
            self._grid.update(self._lidar_points, state[:2])
            occ_at_robot = self._grid.get_probability(float(state[0]), float(state[1]))
            if occ_at_robot >= self._security_threshold:
                in_inflated = True
                escape_target = self._find_escape_target(self._grid, state[:2])

        # Detect mode transition to reset warm-start and setpoint filter
        prev_security = self._security_mode
        self._security_mode = in_inflated
        if in_inflated and not prev_security:
            self._tracker._prev_u = None
            self._tracker._prev_x = None
            self._setpoint_filtered_xy = None
            self._setpoint_filtered_yaw = None

        # Choose path: escape straight-line or normal A* path
        mpc_path = self._a_star_path
        if in_inflated:
            z = float(self._a_star_path[-1][2]) if self._a_star_path else 0.0
            if escape_target is not None:
                mpc_path = [
                    (float(state[0]), float(state[1]), z),
                    (float(escape_target[0]), float(escape_target[1]), z),
                ]
                self.get_logger().warn(
                    f'[MPC-SECURITY] Robot inside inflated obstacle zone! '
                    f'occ={occ_at_robot:.3f} >= {self._security_threshold:.2f}. '
                    f'Escape target: ({escape_target[0]:.2f}, {escape_target[1]:.2f})',
                    throttle_duration_sec=0.5,
                )
            else:
                self.get_logger().warn(
                    f'[MPC-SECURITY] Robot inside inflated obstacle zone '
                    f'(occ={occ_at_robot:.3f}) — no free cell found within '
                    f'{self._security_escape_radius:.1f} m, holding A* path.',
                    throttle_duration_sec=0.5,
                )

        # Solve MPC
        result = self._tracker.solve(
            state,
            mpc_path,
            obstacle_points_2d=obs_2d,
        )
        result.security_mode = in_inflated

        self._solve_count    += 1
        self._total_solve_ms += result.solve_time_ms
        if not result.success:
            self._fail_count += 1

        # Publish predicted trajectory
        if result.x_pred is not None:
            pred_path = Path()
            pred_path.header = self._pose.header
            for i in range(len(result.x_pred)):
                p = PoseStamped()
                p.header = self._pose.header
                p.pose.position.x = float(result.x_pred[i, 0])
                p.pose.position.y = float(result.x_pred[i, 1])
                p.pose.position.z = self._pose.pose.position.z
                yaw = float(result.x_pred[i, 2])
                q = self._yaw_to_quat(yaw)
                p.pose.orientation.x = q[0]
                p.pose.orientation.y = q[1]
                p.pose.orientation.z = q[2]
                p.pose.orientation.w = q[3]
                pred_path.poses.append(p)
            self._pred_path_pub.publish(pred_path)

            # Find lookahead setpoint — walk predicted trajectory for first
            # state >= lookahead_dist from robot.  Fall back to last A* waypoint
            # near goal so the robot homes in rather than oscillating.
            robot_pos = state[:2]
            lookahead_idx = len(result.x_pred) - 1
            found = False
            for i in range(1, len(result.x_pred)):
                if float(np.linalg.norm(result.x_pred[i, :2] - robot_pos)) >= self._lookahead_dist:
                    lookahead_idx = i
                    found = True
                    break

            if found:
                nxt_xy = result.x_pred[lookahead_idx, :2]
                nxt_yaw = float(result.x_pred[lookahead_idx, 2])
            else:
                # Near goal: steer toward last A* waypoint
                last_wp = self._a_star_path[-1]
                nxt_xy = np.array([float(last_wp[0]), float(last_wp[1])])
                nxt_yaw = self._yaw

            # Stabilize setpoint stream to avoid command jitter from path flicker.
            # Filter XY and yaw together so they remain consistent.
            nxt_xy = np.asarray(nxt_xy, dtype=float)
            if self._setpoint_filtered_xy is None:
                self._setpoint_filtered_xy = nxt_xy.copy()
                self._setpoint_filtered_yaw = nxt_yaw
            else:
                jump = nxt_xy - self._setpoint_filtered_xy
                jump_norm = float(np.linalg.norm(jump))
                if self._setpoint_max_step > 0.0 and jump_norm > self._setpoint_max_step:
                    nxt_xy = self._setpoint_filtered_xy + (jump / (jump_norm + 1e-9)) * self._setpoint_max_step
                alpha = np.clip(self._setpoint_alpha, 0.0, 1.0)
                self._setpoint_filtered_xy = (
                    (1.0 - alpha) * self._setpoint_filtered_xy + alpha * nxt_xy
                )
                # Wrap-aware yaw filter
                yaw_err = math.atan2(math.sin(nxt_yaw - self._setpoint_filtered_yaw),
                                     math.cos(nxt_yaw - self._setpoint_filtered_yaw))
                self._setpoint_filtered_yaw = self._setpoint_filtered_yaw + alpha * yaw_err
            nxt_xy = self._setpoint_filtered_xy
            nxt_yaw = self._setpoint_filtered_yaw

            setpoint = PoseStamped()
            setpoint.header = self._pose.header
            setpoint.pose.position.x = float(nxt_xy[0])
            setpoint.pose.position.y = float(nxt_xy[1])
            setpoint.pose.position.z = self._pose.pose.position.z
            q = self._yaw_to_quat(nxt_yaw)
            setpoint.pose.orientation.x = q[0]
            setpoint.pose.orientation.y = q[1]
            setpoint.pose.orientation.z = q[2]
            setpoint.pose.orientation.w = q[3]
            self._setpoint_pub.publish(setpoint)

            self.get_logger().info(
                f'[MPC] #{self._solve_count:04d} '
                f'ok={result.success} '
                f'cost={result.cost:8.1f} '
                f'solve={result.solve_time_ms:5.1f} ms '
                f'avg={self._total_solve_ms / self._solve_count:5.1f} ms  '
                f'fails={self._fail_count}  '
                f'security={self._security_mode}  '
                f'path_wpts={len(self._a_star_path)}(raw={self._a_star_path_raw_len})  '
                f'lookahead_k={lookahead_idx}  '
                f'robot=[{state[0]:.2f},{state[1]:.2f}] '
                f'setpt=[{nxt_xy[0]:.2f}, {nxt_xy[1]:.2f}]',
                throttle_duration_sec=0.5,
            )

        # Publish diagnostics
        diag = Float64MultiArray()
        diag.data = [
            float(1 if result.success else 0),
            result.cost,
            result.solve_time_ms,
            float(self._total_solve_ms / max(self._solve_count, 1)),
            float(self._fail_count),
            float(1 if result.security_mode else 0),   # [5] security protocol active
        ]
        self._diagnostics_pub.publish(diag)

    def _yaw_to_quat(self, yaw: float) -> tuple:
        """Convert yaw angle to quaternion (roll=0, pitch=0, yaw=yaw)."""
        half_yaw = yaw / 2.0
        cy = math.cos(half_yaw)
        sy = math.sin(half_yaw)
        return (0.0, 0.0, sy, cy)  # (qx, qy, qz, qw)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
