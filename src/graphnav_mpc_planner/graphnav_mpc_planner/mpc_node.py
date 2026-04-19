"""
MPC tracker node for graphnav_mpc_planner.

Subscribes:
  ~/path                   nav_msgs/Path          -- planned graph path (from planner_node)
  ~/odom                   nav_msgs/Odometry      -- robot pose + velocity
  /lidar/points_filtered   sensor_msgs/PointCloud2 -- filtered 3-D LiDAR points (world frame)

Publishes:
  /mpc/predicted_path      nav_msgs/Path             -- N-step MPC predicted trajectory
  /mpc/next_setpoint       geometry_msgs/PoseStamped -- lookahead setpoint -> setpoint_to_cmd_vel
  /mpc/diagnostics         std_msgs/Float64MultiArray -- [success, cost, solve_ms, avg_ms, fails, security]

Differences from a_star_mpc_planner/mpc_node.py:
  - Subscribes to ~/path (namespaced, from graphnav planner) instead of /a_star/path
  - Gets robot pose from nav_msgs/Odometry instead of geometry_msgs/PoseStamped
  - Path is in the global graph frame (world/odom) rather than a local occupancy grid frame

author: Lorenzo Ortolani
"""

import math
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray

from graphnav_mpc_planner.gaussian_grid_map import FixedGaussianGridMap
from graphnav_mpc_planner.mpc_tracker import MPCConfig, MPCTracker


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _yaw_to_quat(yaw: float) -> tuple:
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))  # (qx, qy, qz, qw)


class MPCNode(Node):

    def __init__(self):
        super().__init__('graphnav_mpc_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('mpc_N',                    30)
        self.declare_parameter('mpc_dt',                 0.05)
        self.declare_parameter('mpc_vx_max',              1.0)
        self.declare_parameter('mpc_vy_max',              0.5)
        self.declare_parameter('mpc_omega_max',           1.5)
        self.declare_parameter('mpc_v_ref',               0.5)
        self.declare_parameter('mpc_Q_xy',              200.0)
        self.declare_parameter('mpc_Q_yaw',               0.3)
        self.declare_parameter('mpc_Q_terminal',         50.0)
        self.declare_parameter('mpc_R_vel',              10.0)
        self.declare_parameter('mpc_R_omega',             0.5)
        self.declare_parameter('mpc_R_jerk',              1.0)
        self.declare_parameter('mpc_W_obs_sigmoid',      50.0)
        self.declare_parameter('mpc_obs_alpha',           6.0)
        self.declare_parameter('mpc_obs_r',              0.75)
        self.declare_parameter('mpc_max_obs_constraints',  12)
        self.declare_parameter('mpc_obs_check_radius',    3.0)
        self.declare_parameter('mpc_max_iter',            100)
        self.declare_parameter('mpc_warm_start',         True)
        self.declare_parameter('mpc_rate_hz',            10.0)
        self.declare_parameter('mpc_lookahead_dist',      0.8)
        self.declare_parameter('max_lidar_range',         6.0)
        self.declare_parameter('mpc_path_resample_ds',  0.15)
        self.declare_parameter('mpc_path_smooth_window',    7)
        self.declare_parameter('mpc_setpoint_alpha',     0.20)
        self.declare_parameter('mpc_setpoint_max_step',  0.20)
        self.declare_parameter('mpc_setpoint_reset_dist',1.25)
        self.declare_parameter('grid_reso',              0.25)
        self.declare_parameter('grid_half_width',         5.0)
        self.declare_parameter('grid_std',               0.30)
        self.declare_parameter('mpc_security_threshold', 0.25)
        self.declare_parameter('mpc_security_escape_radius', 3.0)

        # ── Build MPCConfig ───────────────────────────────────────────
        cfg = MPCConfig(
            N                   = int(self.get_parameter('mpc_N').value),
            dt                  = float(self.get_parameter('mpc_dt').value),
            vx_max              = float(self.get_parameter('mpc_vx_max').value),
            vy_max              = float(self.get_parameter('mpc_vy_max').value),
            omega_max           = float(self.get_parameter('mpc_omega_max').value),
            v_ref               = float(self.get_parameter('mpc_v_ref').value),
            Q_xy                = float(self.get_parameter('mpc_Q_xy').value),
            Q_yaw               = float(self.get_parameter('mpc_Q_yaw').value),
            Q_terminal          = float(self.get_parameter('mpc_Q_terminal').value),
            R_vel               = float(self.get_parameter('mpc_R_vel').value),
            R_omega             = float(self.get_parameter('mpc_R_omega').value),
            R_jerk              = float(self.get_parameter('mpc_R_jerk').value),
            W_obs_sigmoid       = float(self.get_parameter('mpc_W_obs_sigmoid').value),
            obs_alpha           = float(self.get_parameter('mpc_obs_alpha').value),
            obs_r               = float(self.get_parameter('mpc_obs_r').value),
            max_obs_constraints = int(self.get_parameter('mpc_max_obs_constraints').value),
            obs_check_radius    = float(self.get_parameter('mpc_obs_check_radius').value),
            max_iter            = int(self.get_parameter('mpc_max_iter').value),
            warm_start          = bool(self.get_parameter('mpc_warm_start').value),
        )
        self._tracker = MPCTracker(config=cfg)

        # ── Security protocol ─────────────────────────────────────────
        self._security_threshold     = float(self.get_parameter('mpc_security_threshold').value)
        self._security_escape_radius = float(self.get_parameter('mpc_security_escape_radius').value)
        self._grid = FixedGaussianGridMap(
            reso       = float(self.get_parameter('grid_reso').value),
            half_width = float(self.get_parameter('grid_half_width').value),
            std        = float(self.get_parameter('grid_std').value),
        )
        self._security_mode = False

        # ── Misc settings ─────────────────────────────────────────────
        self._max_lidar_range    = float(self.get_parameter('max_lidar_range').value)
        self._lookahead_dist     = float(self.get_parameter('mpc_lookahead_dist').value)
        self._path_resample_ds   = float(self.get_parameter('mpc_path_resample_ds').value)
        self._path_smooth_window = int(self.get_parameter('mpc_path_smooth_window').value)
        self._setpoint_alpha     = float(self.get_parameter('mpc_setpoint_alpha').value)
        self._setpoint_max_step  = float(self.get_parameter('mpc_setpoint_max_step').value)
        self._setpoint_reset_dist= float(self.get_parameter('mpc_setpoint_reset_dist').value)

        # ── State ─────────────────────────────────────────────────────
        self._odom: Odometry | None           = None
        self._yaw: float                      = 0.0
        self._graph_path: list | None         = None
        self._graph_path_raw_len: int         = 0
        self._lidar_points: np.ndarray | None = None
        self._setpoint_filtered_xy: np.ndarray | None = None
        self._setpoint_filtered_yaw: float | None     = None

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
        self.create_subscription(Odometry,    '~/odom',                    self._odom_cb,  10)
        self.create_subscription(PointCloud2, '/lidar/points_filtered',    self._lidar_cb, sensor_qos)
        self.create_subscription(Path,        '~/path',                    self._path_cb,  10)

        # ── Publishers ────────────────────────────────────────────────
        self._pred_path_pub   = self.create_publisher(Path,                '/mpc/predicted_path', 10)
        self._setpoint_pub    = self.create_publisher(PoseStamped,         '/mpc/next_setpoint',  10)
        self._diagnostics_pub = self.create_publisher(Float64MultiArray,   '/mpc/diagnostics',    10)

        # ── Solve timer ───────────────────────────────────────────────
        rate = float(self.get_parameter('mpc_rate_hz').value)
        self.create_timer(1.0 / rate, self._solve_cb)

        self.get_logger().info('GraphNav MPC node ready')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._odom = msg
        q = msg.pose.pose.orientation
        self._yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

    def _lidar_cb(self, msg: PointCloud2):
        try:
            points = list(point_cloud2.read_points(msg, skip_nans=True))
            if points:
                arr = np.array([(p[0], p[1], p[2]) for p in points], dtype=float)
                if self._odom is not None:
                    px = self._odom.pose.pose.position.x
                    py = self._odom.pose.pose.position.y
                    dists = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
                    arr = arr[dists < self._max_lidar_range]
                self._lidar_points = arr if len(arr) > 0 else None
            else:
                self._lidar_points = None
        except Exception as e:
            self.get_logger().warning(f'LiDAR parse error: {e}')

    def _path_cb(self, msg: Path):
        if msg.poses:
            raw_path = [
                (p.pose.position.x, p.pose.position.y, p.pose.position.z)
                for p in msg.poses
            ]
            self._graph_path_raw_len = len(raw_path)
            smoothed = self._smooth_resample_path(raw_path)
            self._graph_path = smoothed

            # Reset setpoint filter only on large topological jumps
            did_reset = False
            if self._setpoint_filtered_xy is None or self._setpoint_filtered_yaw is None:
                did_reset = True
            else:
                idx = 1 if len(smoothed) > 1 else 0
                anchor_xy = np.array([float(smoothed[idx][0]), float(smoothed[idx][1])], dtype=float)
                if float(np.linalg.norm(anchor_xy - self._setpoint_filtered_xy)) > self._setpoint_reset_dist:
                    did_reset = True

            if did_reset:
                self._setpoint_filtered_xy  = None
                self._setpoint_filtered_yaw = None

            self.get_logger().info(
                f'[GraphNav-MPC] New path: {len(raw_path)} pts → {len(smoothed)} resampled '
                f'| filter_reset={did_reset}',
                throttle_duration_sec=1.0,
            )
        else:
            self._graph_path            = None
            self._graph_path_raw_len    = 0
            self._setpoint_filtered_xy  = None
            self._setpoint_filtered_yaw = None

    # ── MPC solve ─────────────────────────────────────────────────────

    def _solve_cb(self):
        if self._odom is None or self._graph_path is None:
            self.get_logger().warning(
                '[GraphNav-MPC] Waiting for odom and path…', throttle_duration_sec=5.0
            )
            return

        state = np.array([
            self._odom.pose.pose.position.x,
            self._odom.pose.pose.position.y,
            self._yaw,
        ])

        obs_2d = None
        if self._lidar_points is not None and len(self._lidar_points) > 0:
            obs_2d = self._lidar_points[:, :2]

        # ── Security protocol ────────────────────────────────────────
        in_inflated   = False
        escape_target: Optional[np.ndarray] = None
        occ_at_robot  = 0.0
        if self._lidar_points is not None and len(self._lidar_points) > 0:
            self._grid.update(self._lidar_points, state[:2])
            occ_at_robot = self._grid.get_probability(float(state[0]), float(state[1]))
            if occ_at_robot >= self._security_threshold:
                in_inflated   = True
                escape_target = self._find_escape_target(self._grid, state[:2])

        prev_security      = self._security_mode
        self._security_mode = in_inflated
        if in_inflated and not prev_security:
            self._tracker._prev_u       = None
            self._tracker._prev_x       = None
            self._setpoint_filtered_xy  = None
            self._setpoint_filtered_yaw = None

        # Choose path
        mpc_path = self._graph_path
        if in_inflated:
            z = float(self._graph_path[-1][2]) if self._graph_path else 0.0
            if escape_target is not None:
                mpc_path = [
                    (float(state[0]), float(state[1]), z),
                    (float(escape_target[0]), float(escape_target[1]), z),
                ]
                self.get_logger().warning(
                    f'[MPC-SECURITY] Inflated obstacle! occ={occ_at_robot:.3f}. '
                    f'Escape→({escape_target[0]:.2f},{escape_target[1]:.2f})',
                    throttle_duration_sec=0.5,
                )
            else:
                self.get_logger().warning(
                    f'[MPC-SECURITY] Inflated obstacle occ={occ_at_robot:.3f} — '
                    'no free cell nearby, holding graph path.',
                    throttle_duration_sec=0.5,
                )

        # Solve
        result = self._tracker.solve(state, mpc_path, obstacle_points_2d=obs_2d)
        result.security_mode = in_inflated
        self._solve_count    += 1
        self._total_solve_ms += result.solve_time_ms
        if not result.success:
            self._fail_count += 1

        # Publish predicted path
        if result.x_pred is not None:
            pred_path = Path()
            pred_path.header = self._odom.header
            for i in range(len(result.x_pred)):
                ps = PoseStamped()
                ps.header = self._odom.header
                ps.pose.position.x = float(result.x_pred[i, 0])
                ps.pose.position.y = float(result.x_pred[i, 1])
                ps.pose.position.z = self._odom.pose.pose.position.z
                q = _yaw_to_quat(float(result.x_pred[i, 2]))
                ps.pose.orientation.x = q[0]
                ps.pose.orientation.y = q[1]
                ps.pose.orientation.z = q[2]
                ps.pose.orientation.w = q[3]
                pred_path.poses.append(ps)
            self._pred_path_pub.publish(pred_path)

            # Lookahead setpoint selection
            robot_pos   = state[:2]
            lookahead_k = len(result.x_pred) - 1
            found       = False
            for i in range(1, len(result.x_pred)):
                if float(np.linalg.norm(result.x_pred[i, :2] - robot_pos)) >= self._lookahead_dist:
                    lookahead_k = i
                    found       = True
                    break

            if found:
                nxt_xy  = result.x_pred[lookahead_k, :2]
                nxt_yaw = float(result.x_pred[lookahead_k, 2])
            else:
                last_wp = self._graph_path[-1]
                nxt_xy  = np.array([float(last_wp[0]), float(last_wp[1])])
                nxt_yaw = self._yaw

            # Stabilise setpoint
            nxt_xy = np.asarray(nxt_xy, dtype=float)
            if self._setpoint_filtered_xy is None:
                self._setpoint_filtered_xy  = nxt_xy.copy()
                self._setpoint_filtered_yaw = nxt_yaw
            else:
                jump      = nxt_xy - self._setpoint_filtered_xy
                jump_norm = float(np.linalg.norm(jump))
                if self._setpoint_max_step > 0.0 and jump_norm > self._setpoint_max_step:
                    nxt_xy = self._setpoint_filtered_xy + (jump / (jump_norm + 1e-9)) * self._setpoint_max_step
                alpha = float(np.clip(self._setpoint_alpha, 0.0, 1.0))
                self._setpoint_filtered_xy = (1.0 - alpha) * self._setpoint_filtered_xy + alpha * nxt_xy
                yaw_err = math.atan2(
                    math.sin(nxt_yaw - self._setpoint_filtered_yaw),
                    math.cos(nxt_yaw - self._setpoint_filtered_yaw),
                )
                self._setpoint_filtered_yaw = self._setpoint_filtered_yaw + alpha * yaw_err

            nxt_xy  = self._setpoint_filtered_xy
            nxt_yaw = self._setpoint_filtered_yaw

            sp = PoseStamped()
            sp.header = self._odom.header
            sp.pose.position.x = float(nxt_xy[0])
            sp.pose.position.y = float(nxt_xy[1])
            sp.pose.position.z = self._odom.pose.pose.position.z
            q = _yaw_to_quat(nxt_yaw)
            sp.pose.orientation.x = q[0]
            sp.pose.orientation.y = q[1]
            sp.pose.orientation.z = q[2]
            sp.pose.orientation.w = q[3]
            self._setpoint_pub.publish(sp)

            self.get_logger().info(
                f'[GraphNav-MPC] #{self._solve_count:04d} '
                f'ok={result.success} cost={result.cost:8.1f} '
                f'solve={result.solve_time_ms:5.1f}ms '
                f'avg={self._total_solve_ms / self._solve_count:5.1f}ms '
                f'fails={self._fail_count} security={self._security_mode} '
                f'path_wpts={len(self._graph_path)}(raw={self._graph_path_raw_len}) '
                f'lookahead_k={lookahead_k} '
                f'robot=[{state[0]:.2f},{state[1]:.2f}] '
                f'setpt=[{nxt_xy[0]:.2f},{nxt_xy[1]:.2f}]',
                throttle_duration_sec=0.5,
            )

        # Diagnostics
        diag      = Float64MultiArray()
        diag.data = [
            float(1 if result.success else 0),
            result.cost,
            result.solve_time_ms,
            float(self._total_solve_ms / max(self._solve_count, 1)),
            float(self._fail_count),
            float(1 if result.security_mode else 0),
        ]
        self._diagnostics_pub.publish(diag)

    # ── Helpers ───────────────────────────────────────────────────────

    def _smooth_resample_path(self, path_xyz: list) -> list:
        if not path_xyz or len(path_xyz) < 2:
            return path_xyz
        xy = np.array([(p[0], p[1]) for p in path_xyz], dtype=float)

        # Remove duplicate consecutive points
        dxy  = np.diff(xy, axis=0)
        keep = np.hstack(([True], np.linalg.norm(dxy, axis=1) > 1e-4))
        xy   = xy[keep]
        if len(xy) < 2:
            z = float(path_xyz[-1][2])
            return [(float(xy[0, 0]), float(xy[0, 1]), z)]

        seg   = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc   = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(arc[-1])
        if total <= 1e-6:
            z = float(path_xyz[-1][2])
            return [(float(xy[0, 0]), float(xy[0, 1]), z), (float(xy[-1, 0]), float(xy[-1, 1]), z)]

        ds = max(self._path_resample_ds, 1e-2)
        s  = np.arange(0.0, total + 1e-9, ds)
        if s[-1] < total:
            s = np.append(s, total)
        x = np.interp(s, arc, xy[:, 0])
        y = np.interp(s, arc, xy[:, 1])
        pts = np.column_stack((x, y))

        win = self._path_smooth_window
        if win >= 3 and (win % 2 == 1) and len(pts) >= win:
            kernel  = np.ones(win, dtype=float) / float(win)
            xs = np.convolve(pts[:, 0], kernel, mode='same')
            ys = np.convolve(pts[:, 1], kernel, mode='same')
            pts[:, 0] = xs
            pts[:, 1] = ys
            pts[0]    = xy[0]
            pts[-1]   = xy[-1]

        z = float(path_xyz[-1][2])
        return [(float(p[0]), float(p[1]), z) for p in pts]

    def _find_escape_target(
        self,
        grid: FixedGaussianGridMap,
        robot_xy: np.ndarray,
    ) -> Optional[np.ndarray]:
        ix0, iy0 = grid.world_to_index(float(robot_xy[0]), float(robot_xy[1]))
        if ix0 is None:
            return None
        visited: set = set()
        queue: deque  = deque()
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


# ─────────────────────────────────────────────────────────────────────────────

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
