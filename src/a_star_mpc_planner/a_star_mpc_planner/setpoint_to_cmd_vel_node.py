"""
setpoint_to_cmd_vel_node.py

Converts MPC lookahead setpoints into body-frame /cmd_vel commands for CHAMP/Gazebo.

Subscribes:
  /go2/pose          geometry_msgs/PoseStamped
  /mpc/next_setpoint geometry_msgs/PoseStamped

Publishes:
  /cmd_vel           geometry_msgs/Twist

Behavior:
  - Proportional XY controller in robot body frame.
  - Optional yaw controller (disabled by default to avoid spin-in-place).
  - Safety timeout: publishes zero if setpoint stream is stale.
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


class SetpointToCmdVelNode(Node):

    def __init__(self):
        super().__init__('setpoint_to_cmd_vel_node')

        self.declare_parameter('cmd_rate_hz', 20.0)
        self.declare_parameter('cmd_kp_xy', 1.0)
        self.declare_parameter('cmd_kp_yaw', 1.5)
        self.declare_parameter('cmd_max_vx', 0.8)
        self.declare_parameter('cmd_max_vy', 0.4)
        self.declare_parameter('cmd_max_omega', 1.2)
        self.declare_parameter('cmd_stop_radius', 0.2)
        self.declare_parameter('setpoint_timeout_sec', 1.0)
        self.declare_parameter('enable_yaw_control', False)

        self._rate_hz = float(self.get_parameter('cmd_rate_hz').value)
        self._kp_xy = float(self.get_parameter('cmd_kp_xy').value)
        self._kp_yaw = float(self.get_parameter('cmd_kp_yaw').value)
        self._max_vx = float(self.get_parameter('cmd_max_vx').value)
        self._max_vy = float(self.get_parameter('cmd_max_vy').value)
        self._max_omega = float(self.get_parameter('cmd_max_omega').value)
        self._stop_radius = float(self.get_parameter('cmd_stop_radius').value)
        self._setpoint_timeout = float(self.get_parameter('setpoint_timeout_sec').value)
        self._enable_yaw_control = bool(self.get_parameter('enable_yaw_control').value)

        self._pose: PoseStamped | None = None
        self._yaw = 0.0
        self._setpoint: PoseStamped | None = None
        self._setpoint_rx_time = None

        self.create_subscription(PoseStamped, '/go2/pose', self._pose_cb, 10)
        self.create_subscription(PoseStamped, '/mpc/next_setpoint', self._setpoint_cb, 10)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(1.0 / self._rate_hz, self._control_cb)

        self.get_logger().info(
            'setpoint_to_cmd_vel ready: /mpc/next_setpoint + /go2/pose -> /cmd_vel'
        )

    def _pose_cb(self, msg: PoseStamped):
        self._pose = msg
        q = msg.pose.orientation
        self._yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

    def _setpoint_cb(self, msg: PoseStamped):
        self._setpoint = msg
        self._setpoint_rx_time = self.get_clock().now()

    def _publish_zero(self):
        self._cmd_pub.publish(Twist())

    def _control_cb(self):
        if self._pose is None or self._setpoint is None or self._setpoint_rx_time is None:
            self._publish_zero()
            return

        now = self.get_clock().now()
        age_sec = (now - self._setpoint_rx_time).nanoseconds * 1e-9
        if age_sec > self._setpoint_timeout:
            self.get_logger().warn(
                f'setpoint timeout ({age_sec:.2f}s > {self._setpoint_timeout:.2f}s), zeroing /cmd_vel',
                throttle_duration_sec=1.0,
            )
            self._publish_zero()
            return

        px = float(self._pose.pose.position.x)
        py = float(self._pose.pose.position.y)
        sx = float(self._setpoint.pose.position.x)
        sy = float(self._setpoint.pose.position.y)

        dx_world = sx - px
        dy_world = sy - py
        dist = math.hypot(dx_world, dy_world)

        if dist <= self._stop_radius:
            self._publish_zero()
            return

        # World -> robot body frame (x forward, y left)
        ex = math.cos(self._yaw) * dx_world + math.sin(self._yaw) * dy_world
        ey = -math.sin(self._yaw) * dx_world + math.cos(self._yaw) * dy_world

        cmd = Twist()
        cmd.linear.x = _clamp(self._kp_xy * ex, -self._max_vx, self._max_vx)
        cmd.linear.y = _clamp(self._kp_xy * ey, -self._max_vy, self._max_vy)
        if self._enable_yaw_control:
            # Use MPC-predicted yaw stored in setpoint orientation (path-aligned heading).
            q = self._setpoint.pose.orientation
            yaw_sp = _quat_to_yaw(q.x, q.y, q.z, q.w)
            yaw_err = _wrap_to_pi(yaw_sp - self._yaw)
            cmd.angular.z = _clamp(self._kp_yaw * yaw_err, -self._max_omega, self._max_omega)
        else:
            cmd.angular.z = 0.0

        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SetpointToCmdVelNode()
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
