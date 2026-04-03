"""
MPC tracker ROS2 node for the Go2 quadruped robot.

Architecture
------------
  Subscribes:
    /go2/pose    PoseStamped      — robot pose (position + orientation)
    /lidar/points_filtered PointCloud2     — 3-D lidar hits (world frame)
    /wildos/path nav_msgs/Path    — WildOS graph planner path

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
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray

from mpc.mpc_tracker import MPCConfig, MPCTracker


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
        self.declare_parameter('mpc_N',            50)
        self.declare_parameter('mpc_dt',           0.1)
        self.declare_parameter('mpc_vx_max',       0.3)
        self.declare_parameter('mpc_vy_max',       0.25)
        self.declare_parameter('mpc_omega_max',    0.5)
        self.declare_parameter('mpc_v_ref',        0.25)
        self.declare_parameter('mpc_Q_xy',        200.0)
        self.declare_parameter('mpc_Q_yaw',        1.0)
        self.declare_parameter('mpc_Q_terminal',  100.0)
        self.declare_parameter('mpc_R_vel',        1.0)
        self.declare_parameter('mpc_R_omega',      0.5)
        self.declare_parameter('mpc_R_jerk',       0.5)
        self.declare_parameter('mpc_W_obs_sigmoid',       2.0)
        self.declare_parameter('mpc_obs_alpha',           4.0)
        self.declare_parameter('mpc_obs_r',               0.25)
        self.declare_parameter('mpc_max_obs_constraints', 15)
        self.declare_parameter('mpc_obs_check_radius',    1.5)
        self.declare_parameter('mpc_max_iter',   100)
        self.declare_parameter('mpc_warm_start',  True)
        self.declare_parameter('mpc_rate_hz',      20.0)
        self.declare_parameter('mpc_lookahead_dist', 0.5)
        self.declare_parameter('max_lidar_range',  20.0)
        self.declare_parameter('mpc_path_resample_ds', 0.20)
        self.declare_parameter('mpc_path_smooth_window', 5)
        self.declare_parameter('mpc_setpoint_alpha', 0.35)
        self.declare_parameter('mpc_setpoint_max_step', 0.30)

        # Direct cmd_vel from MPC optimal control u_opt[0]
        self.declare_parameter('cmd_rate_hz',       50.0)
        self.declare_parameter('cmd_max_vx',         0.3)
        self.declare_parameter('cmd_max_vy',         0.25)
        self.declare_parameter('cmd_max_omega',      0.5)
        self.declare_parameter('cmd_stop_radius',    0.2)
        self.declare_parameter('setpoint_timeout_sec', 2.0)

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

        self._max_lidar_range = float(self.get_parameter('max_lidar_range').value)
        self._lookahead_dist  = float(self.get_parameter('mpc_lookahead_dist').value)
        self._path_resample_ds = float(self.get_parameter('mpc_path_resample_ds').value)
        self._path_smooth_window = int(self.get_parameter('mpc_path_smooth_window').value)
        self._setpoint_alpha = float(self.get_parameter('mpc_setpoint_alpha').value)
        self._setpoint_max_step = float(self.get_parameter('mpc_setpoint_max_step').value)

        # ── State ─────────────────────────────────────────────────────
        self._pose: PoseStamped | None = None
        self._yaw = 0.0
        self._wildos_path: list | None = None
        self._wildos_path_raw_len = 0
        self._lidar_points: np.ndarray | None = None
        self._setpoint_filtered_xy: np.ndarray | None = None
        self._setpoint_filtered_yaw: float | None = None

        # Direct cmd_vel from MPC u_opt[0] (receding horizon control)
        self._cmd_max_vx    = float(self.get_parameter('cmd_max_vx').value)
        self._cmd_max_vy    = float(self.get_parameter('cmd_max_vy').value)
        self._cmd_max_omega = float(self.get_parameter('cmd_max_omega').value)
        self._cmd_stop_radius = float(self.get_parameter('cmd_stop_radius').value)
        self._cmd_timeout   = float(self.get_parameter('setpoint_timeout_sec').value)
        self._last_u0: np.ndarray | None = None   # [vx, vy, omega] body frame
        self._last_u0_time = None

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
        self.create_subscription(Path,        '/wildos/path',     self._path_cb,  10)

        # ── Publishers ────────────────────────────────────────────────
        self._pred_path_pub = self.create_publisher(Path,                '/mpc/predicted_path', 10)
        self._setpoint_pub  = self.create_publisher(PoseStamped,         '/mpc/next_setpoint',  10)
        self._diagnostics_pub = self.create_publisher(Float64MultiArray, '/mpc/diagnostics',    10)
        self._cmd_vel_pub   = self.create_publisher(Twist,               '/cmd_vel',            10)

        # ── Timers ────────────────────────────────────────────────────
        rate = float(self.get_parameter('mpc_rate_hz').value)
        self.create_timer(1.0 / rate, self._solve_cb)

        # cmd_vel is republished at cmd_rate_hz (holds last u_opt[0] between solves)
        cmd_rate = float(self.get_parameter('cmd_rate_hz').value)
        self.create_timer(1.0 / cmd_rate, self._cmd_vel_cb)

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

    def _cmd_vel_cb(self):
        """
        Publish /cmd_vel at cmd_rate_hz using the last MPC-optimal u_opt[0].

        This implements receding-horizon control: the first optimal control action
        (already guaranteed safe by the MPC hard obstacle constraints) is applied
        directly to the robot, bypassing any P-controller approximation.
        Safety fallback: publishes zero Twist if MPC has not solved recently.
        """
        cmd = Twist()

        if self._last_u0 is not None and self._last_u0_time is not None:
            age = (self.get_clock().now() - self._last_u0_time).nanoseconds * 1e-9
            if age <= self._cmd_timeout:
                cmd.linear.x  = float(np.clip(self._last_u0[0], -self._cmd_max_vx,    self._cmd_max_vx))
                cmd.linear.y  = float(np.clip(self._last_u0[1], -self._cmd_max_vy,    self._cmd_max_vy))
                cmd.angular.z = float(np.clip(self._last_u0[2], -self._cmd_max_omega, self._cmd_max_omega))
            else:
                self.get_logger().warn(
                    f'MPC cmd_vel timeout ({age:.2f}s), zeroing /cmd_vel',
                    throttle_duration_sec=1.0,
                )

        self._cmd_vel_pub.publish(cmd)

    def _path_cb(self, msg: Path):
        """
        Store WildOS path and keep the setpoint filter stable across frequent replans.
        
        New path means WildOS planner has replanned based on robot's current position and sensors.
        We only reset the filtered setpoint when the new plan jumps far away from the
        current filtered command; otherwise frequent replans can make the command
        stream keep restarting near the robot and appear stuck.
        """
        if msg.poses:
            raw_path = [
                (p.pose.position.x, p.pose.position.y, p.pose.position.z)
                for p in msg.poses
            ]
            self._wildos_path_raw_len = len(raw_path)
            self._wildos_path = self._smooth_resample_path(raw_path)

            if self._setpoint_filtered_xy is not None and self._wildos_path:
                first_xy = np.asarray(self._wildos_path[0][:2], dtype=float)
                filter_jump = float(np.linalg.norm(first_xy - self._setpoint_filtered_xy))
                if filter_jump > 1.0:
                    self.get_logger().info(
                        f'[MPC] Resetting setpoint filter due path jump ({filter_jump:.2f} m)',
                        throttle_duration_sec=1.0,
                    )
                    self._setpoint_filtered_xy = None
                    self._setpoint_filtered_yaw = None

            self.get_logger().info(
                f'[MPC] Received NEW WildOS path: {len(raw_path)} raw points → '
                f'{len(self._wildos_path)} resampled points',
                throttle_duration_sec=1.0,
            )
        else:
            self._wildos_path = None
            self._wildos_path_raw_len = 0
            self._setpoint_filtered_xy = None
            self._setpoint_filtered_yaw = None

    def _smooth_resample_path(self, path_xyz: list) -> list:
        """
        Resample and smooth the WildOS path before feeding it to MPC.
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

    def _solve_cb(self):
        """Solve MPC and publish results."""
        if self._pose is None or self._wildos_path is None:
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

        # Solve MPC
        result = self._tracker.solve(
            state,
            self._wildos_path,
            obstacle_points_2d=obs_2d,
        )

        self._solve_count    += 1
        self._total_solve_ms += result.solve_time_ms
        if not result.success:
            self._fail_count += 1

        # ── Direct MPC control: store u_opt[0] for _cmd_vel_cb ───────
        # Receding-horizon principle: apply the first optimal control action
        # directly.  u_opt is in body frame [vx, vy, omega] so it maps 1:1
        # to cmd_vel without any P-controller approximation.
        if result.success and result.u_opt is not None and len(result.u_opt) > 0:
            # Stop near the path endpoint to avoid oscillation at goal.
            near_goal = False
            if self._wildos_path and self._pose is not None:
                last_xy = np.array(self._wildos_path[-1][:2], dtype=float)
                robot_xy = state[:2]
                if float(np.linalg.norm(last_xy - robot_xy)) <= self._cmd_stop_radius:
                    near_goal = True
            self._last_u0 = np.zeros(3) if near_goal else result.u_opt[0].copy()
        else:
            # Zero on failure — safety first.
            self._last_u0 = np.zeros(3)
        self._last_u0_time = self.get_clock().now()

        # Publish predicted trajectory.  If the solver returns an invalid
        # trajectory, fall back to the current planned path so RViz never sees
        # a silent gap.
        pred_xy = result.x_pred
        if pred_xy is None or len(pred_xy) == 0:
            self.get_logger().warn('[MPC] Solver returned no predicted trajectory; predicted_path not updated', throttle_duration_sec=2.0)
        else:
            pred_path = Path()
            pred_path.header = self._pose.header
            for i in range(len(pred_xy)):
                p = PoseStamped()
                p.header = self._pose.header
                p.pose.position.x = float(pred_xy[i, 0])
                p.pose.position.y = float(pred_xy[i, 1])
                p.pose.position.z = self._pose.pose.position.z
                yaw = float(pred_xy[i, 2])
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
            lookahead_idx = len(pred_xy) - 1
            found = False
            for i in range(1, len(pred_xy)):
                if float(np.linalg.norm(pred_xy[i, :2] - robot_pos)) >= self._lookahead_dist:
                    lookahead_idx = i
                    found = True
                    break

            if found:
                nxt_xy = pred_xy[lookahead_idx, :2]
                nxt_yaw = float(pred_xy[lookahead_idx, 2])
            else:
                # MPC prediction never reached lookahead_dist — this happens
                # when obstacle avoidance curves / slows the predicted trajectory
                # so it stays within lookahead_dist, OR when the robot is already
                # very close to the goal.  Use the last predicted MPC state so the
                # setpoint stays on the MPC-planned safe trajectory rather than
                # jumping to the 6 m-away global goal and triggering oscillation.
                nxt_xy = pred_xy[-1, :2]
                nxt_yaw = float(pred_xy[-1, 2])

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
                f'path_wpts={len(self._wildos_path)}(raw={self._wildos_path_raw_len})  '
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
