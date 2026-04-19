"""
GraphNav Dijkstra planner node.

Subscribes:
  ~/nav_graph   graphnav_msgs/NavigationGraph   -- scored navigation graph
  ~/goal_pose   geometry_msgs/PoseStamped       -- navigation goal
  ~/odom        nav_msgs/Odometry               -- robot odometry (used to detect goal reach)

Publishes:
  ~/path        nav_msgs/Path                   -- planned waypoint sequence

The graph planner mirrors graphnav_planner::PlannerNode (C++) in Python,
replacing the C++ PathFollowerNode with MPC downstream.

author: Lorenzo Ortolani
"""

import math

import rclpy
from rclpy.node import Node

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped transform)

from geometry_msgs.msg import PoseStamped
from graphnav_msgs.msg import NavigationGraph
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from graphnav_mpc_planner.graph_planner import GraphNavPlanner


class PlannerNode(Node):

    def __init__(self):
        super().__init__('graphnav_mpc_planner_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('frontier_dist_cost_factor', 2.0)
        self.declare_parameter('goal_dist_cost_factor',     1.0)
        self.declare_parameter('frontier_score_factor',    10.0)
        self.declare_parameter('min_local_frontier_score',  0.4)
        self.declare_parameter('local_frontier_radius',     7.0)
        self.declare_parameter('path_smoothness_period',   10.0)
        self.declare_parameter('trav_class',            'default')
        self.declare_parameter('goal_radius',               3.0)

        # ── Planner ────────────────────────────────────────────────────
        self._planner = GraphNavPlanner(logger=self.get_logger())
        self._planner.frontier_dist_cost_factor = float(
            self.get_parameter('frontier_dist_cost_factor').value)
        self._planner.goal_dist_cost_factor = float(
            self.get_parameter('goal_dist_cost_factor').value)
        self._planner.frontier_score_factor = float(
            self.get_parameter('frontier_score_factor').value)
        self._planner.min_local_frontier_score = float(
            self.get_parameter('min_local_frontier_score').value)
        self._planner.local_frontier_radius = float(
            self.get_parameter('local_frontier_radius').value)
        self._planner.path_smoothness_period = float(
            self.get_parameter('path_smoothness_period').value)
        self._planner.set_trav_class(
            self.get_parameter('trav_class').get_parameter_value().string_value)

        self._goal_radius = float(self.get_parameter('goal_radius').value)

        # ── TF ────────────────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── State ─────────────────────────────────────────────────────
        self._goal_pose: PoseStamped | None = None
        self._odom: Odometry | None = None
        self._latest_graph_header: Header | None = None

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            NavigationGraph, '~/nav_graph', self._graph_cb, 10)
        self.create_subscription(
            PoseStamped, '~/goal_pose', self._goal_cb, 10)
        self.create_subscription(
            Odometry, '~/odom', self._odom_cb, 10)

        # ── Publishers ────────────────────────────────────────────────
        self._path_pub = self.create_publisher(Path, '~/path', 10)

        # ── Debug state ───────────────────────────────────────────────
        self._graph_received = False
        self._goal_received = False
        self._odom_received = False

        self.get_logger().info('GraphNav MPC planner node ready')
        self.get_logger().info('Waiting for topics:')
        self.get_logger().info('  - ~/nav_graph (NavigationGraph)')
        self.get_logger().info('  - ~/goal_pose (PoseStamped)')
        self.get_logger().info('  - ~/odom (Odometry)')
        self.get_logger().info('Use `ros2 topic list` to check available topics')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _graph_cb(self, msg: NavigationGraph):
        if not self._graph_received:
            self.get_logger().info(f'✓ Received navigation graph with {len(msg.nodes)} nodes')
            self._graph_received = True
        self._planner.update_graph(msg)
        self._latest_graph_header = msg.header
        self._replan()

    def _goal_cb(self, msg: PoseStamped):
        if not self._goal_received:
            self.get_logger().info(f'✓ Received goal pose in frame "{msg.header.frame_id}"')
            self._goal_received = True
        self._goal_pose = msg
        self._replan()

    def _odom_cb(self, msg: Odometry):
        if not self._odom_received:
            self.get_logger().info(f'✓ Receiving odometry from frame "{msg.header.frame_id}"')
            self._odom_received = True
        self._odom = msg
        if self._goal_pose is None:
            return
        # Clear goal once robot is within goal_radius
        try:
            odom_pose = PoseStamped()
            odom_pose.header = msg.header
            odom_pose.pose = msg.pose.pose
            goal_in_odom = self._tf_buffer.transform(
                self._goal_pose, msg.header.frame_id, timeout=rclpy.duration.Duration(seconds=0.1)
            )
            dx = goal_in_odom.pose.position.x - msg.pose.pose.position.x
            dy = goal_in_odom.pose.position.y - msg.pose.pose.position.y
            if math.hypot(dx, dy) < self._goal_radius:
                self.get_logger().info('Goal reached — clearing goal')
                self._goal_pose = None
        except Exception:
            pass

    # ── Planning ──────────────────────────────────────────────────────

    def _replan(self):
        if self._goal_pose is None or self._latest_graph_header is None:
            missing = []
            if self._goal_pose is None:
                missing.append('goal_pose')
            if self._latest_graph_header is None:
                missing.append('nav_graph')
            if missing:
                self.get_logger().debug(
                    f'Cannot plan yet — waiting for: {", ".join(missing)}', throttle_duration_sec=5.0
                )
            return

        # Transform goal into the graph coordinate frame
        try:
            goal_stamped = PoseStamped()
            goal_stamped.header.stamp = self._latest_graph_header.stamp
            goal_stamped.header.frame_id = self._goal_pose.header.frame_id
            goal_stamped.pose = self._goal_pose.pose
            goal_in_graph = self._tf_buffer.transform(
                goal_stamped,
                self._latest_graph_header.frame_id,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as ex:
            self.get_logger().warning(
                f'Could not transform goal to graph frame: {ex}', throttle_duration_sec=2.0
            )
            return

        goal_xyz = (
            goal_in_graph.pose.position.x,
            goal_in_graph.pose.position.y,
            goal_in_graph.pose.position.z,
        )

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        path_pts = self._planner.plan_to_goal(goal_xyz, self._goal_radius, now_sec)

        if not path_pts:
            self.get_logger().warning(
                'No path found by graph planner', throttle_duration_sec=5.0
            )
            return

        # ── Build nav_msgs/Path ───────────────────────────────────────
        path_msg = Path()
        path_msg.header = self._latest_graph_header

        for i, (px, py, pz) in enumerate(path_pts):
            ps = PoseStamped()
            ps.header = self._latest_graph_header
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.position.z = pz

            # Orientation: heading toward next waypoint
            if i < len(path_pts) - 1:
                dx = path_pts[i + 1][0] - px
                dy = path_pts[i + 1][1] - py
                yaw = math.atan2(dy, dx)
            else:
                yaw = _quat_to_yaw(
                    goal_in_graph.pose.orientation.x,
                    goal_in_graph.pose.orientation.y,
                    goal_in_graph.pose.orientation.z,
                    goal_in_graph.pose.orientation.w,
                )
            ps.pose.orientation = _yaw_to_quat_msg(yaw)
            path_msg.poses.append(ps)

        self._path_pub.publish(path_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _yaw_to_quat_msg(yaw: float):
    from geometry_msgs.msg import Quaternion
    half = yaw / 2.0
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
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
