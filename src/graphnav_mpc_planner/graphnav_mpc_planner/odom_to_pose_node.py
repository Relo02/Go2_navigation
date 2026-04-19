"""
odom_to_pose_node.py

Republishes nav_msgs/Odometry from /odom as geometry_msgs/PoseStamped on /go2/pose.

This bridge is needed because the graphnav_mpc and other nodes subscribe to /go2/pose
(PoseStamped) but the CHAMP/EKF stack only publishes /odom (Odometry).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class OdomToPoseNode(Node):

    def __init__(self):
        super().__init__('odom_to_pose_node')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub = self.create_publisher(PoseStamped, '/go2/pose', 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, sensor_qos)
        self._msg_count = 0

        self.get_logger().info('odom_to_pose_node ready: /odom → /go2/pose')

    def _odom_cb(self, msg: Odometry):
        self._msg_count += 1

        # Log first message to confirm bridge is alive
        if self._msg_count == 1:
            self.get_logger().info(
                f'✓ First /odom received — publishing /go2/pose'
            )

        # Convert Odometry -> PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose

        self._pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToPoseNode()
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
