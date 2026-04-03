#!/usr/bin/env python3

import math
import uuid

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from graphnav_msgs.msg import Edge
from graphnav_msgs.msg import EdgeTraversability
from graphnav_msgs.msg import NavigationGraph
from graphnav_msgs.msg import Node
from graphnav_msgs.msg import NodeTraversabilityProperties
from graphnav_msgs.msg import UUID
from rclpy.node import Node as RosNode
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class MockNavGraphPublisher(RosNode):
    """Publish a simple connected NavigationGraph for simulation testing.

    This is a fallback source for /nav_graph so the WildOS-style planner can
    run in simulation before a full graph-construction backend is integrated.
    """

    def __init__(self) -> None:
        super().__init__("mock_nav_graph_publisher")

        self.declare_parameter("graph_topic", "/nav_graph")
        self.declare_parameter("pose_topic", "/go2/pose")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("lidar_topic", "/lidar/points_filtered")
        self.declare_parameter("graph_frame_id", "odom")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("far_publish_period_sec", 1.5)
        self.declare_parameter("near_publish_period_sec", 0.4)
        self.declare_parameter("near_obstacle_dist_m", 1.2)
        self.declare_parameter("far_motion_threshold_m", 0.25)
        self.declare_parameter("segment_spacing", 1.5)
        self.declare_parameter("lateral_offset", 1.5)
        self.declare_parameter("max_segments", 60)
        self.declare_parameter("obstacle_influence_radius", 1.40)
        self.declare_parameter("obstacle_block_radius", 0.70)
        self.declare_parameter("obstacle_cost_scale", 20.0)

        graph_topic = str(self.get_parameter("graph_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        goal_topic = str(self.get_parameter("goal_topic").value)
        lidar_topic = str(self.get_parameter("lidar_topic").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._graph_frame_id = str(self.get_parameter("graph_frame_id").value)
        self._far_publish_period_sec = max(float(self.get_parameter("far_publish_period_sec").value), 0.2)
        self._near_publish_period_sec = max(float(self.get_parameter("near_publish_period_sec").value), 0.1)
        self._near_obstacle_dist_m = max(float(self.get_parameter("near_obstacle_dist_m").value), 0.1)
        self._far_motion_threshold_m = max(float(self.get_parameter("far_motion_threshold_m").value), 0.01)

        self._segment_spacing = max(float(self.get_parameter("segment_spacing").value), 0.2)
        self._lateral_offset = max(float(self.get_parameter("lateral_offset").value), 0.0)
        self._max_segments = max(int(self.get_parameter("max_segments").value), 2)
        self._obstacle_influence_radius = max(float(self.get_parameter("obstacle_influence_radius").value), 0.2)
        self._obstacle_block_radius = max(float(self.get_parameter("obstacle_block_radius").value), 0.1)
        # Allow 0.0 to fully disable obstacle penalties in the fallback graph.
        self._obstacle_cost_scale = max(float(self.get_parameter("obstacle_cost_scale").value), 0.0)

        self._graph_pub = self.create_publisher(NavigationGraph, graph_topic, 10)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 10)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)
        self.create_subscription(PointCloud2, lidar_topic, self._lidar_cb, qos_profile_sensor_data)

        self._pose: PoseStamped | None = None
        self._goal: PoseStamped | None = None
        self._lidar_xy: np.ndarray | None = None
        self._last_publish_time = None
        self._last_publish_pose_xy: np.ndarray | None = None
        self._last_goal_xy: np.ndarray | None = None

        self.create_timer(1.0 / max(rate_hz, 0.2), self._publish_graph)
        self.get_logger().info("mock_nav_graph_publisher started")

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._pose = msg

    def _goal_cb(self, msg: PoseStamped) -> None:
        self._goal = msg

    def _lidar_cb(self, msg: PointCloud2) -> None:
        try:
            pts = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if pts:
                self._lidar_xy = np.asarray([(p[0], p[1]) for p in pts], dtype=float)
                # Log periodically to avoid spam
                if len(pts) > 100:
                    self.get_logger().debug(f"Received {len(pts)} LiDAR points")
            else:
                self._lidar_xy = None
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse filtered lidar: {exc}")

    @staticmethod
    def _uuid_msg() -> UUID:
        out = UUID()
        out.id = list(uuid.uuid4().bytes)
        return out

    @staticmethod
    def _make_trav_props(x: float, y: float, frontier_bias: float = 1.0) -> NodeTraversabilityProperties:
        tp = NodeTraversabilityProperties()
        tp.is_frontier = True
        tp.explored_radius = 5.0 * frontier_bias
        tp.free_radius = 5.0 * frontier_bias
        tp.frontier_points = []
        pt = tp.frontier_points
        # Use a tiny frontier segment around the node so the planner has a
        # meaningful frontier geometry to score without depending on a full
        # exploration backend.
        from geometry_msgs.msg import Point
        p0 = Point(); p0.x = float(x); p0.y = float(y); p0.z = 0.0
        p1 = Point(); p1.x = float(x + 0.15); p1.y = float(y); p1.z = 0.0
        pt.append(p0)
        pt.append(p1)
        return tp

    def _make_node(self, x: float, y: float, z: float, frame_id: str, frontier_bias: float = 1.0) -> Node:
        n = Node()
        n.uuid = self._uuid_msg()
        n.pose.position.x = float(x)
        n.pose.position.y = float(y)
        n.pose.position.z = float(z)
        n.pose.orientation.w = 1.0
        n.trav_properties = [self._make_trav_props(x, y, frontier_bias)]
        return n

    @staticmethod
    def _make_edge(from_idx: int, to_idx: int, cost: float) -> Edge:
        e = Edge()
        e.from_idx = from_idx
        e.to_idx = to_idx
        tr = EdgeTraversability()
        tr.traversability_cost = float(max(cost, 0.01))
        e.traversability = [tr]
        return e

    def _node_obstacle_cost(self, x: float, y: float) -> float:
        if self._obstacle_cost_scale <= 1e-6:
            return 0.0

        if self._lidar_xy is None or len(self._lidar_xy) == 0:
            return 0.0

        dx = self._lidar_xy[:, 0] - x
        dy = self._lidar_xy[:, 1] - y
        d = float(np.min(np.hypot(dx, dy)))

        if d <= self._obstacle_block_radius:
            # Return a high cost but not crushing. Dijkstra will avoid but can still route if needed.
            return self._obstacle_cost_scale * 10.0  # Reduced from 100.0

        if d >= self._obstacle_influence_radius:
            return 0.0

        # Smooth penalty that grows as obstacle approaches, but capped to prevent graph collapse.
        ratio = (self._obstacle_influence_radius - d) / max(self._obstacle_influence_radius - self._obstacle_block_radius, 1e-3)
        cost = self._obstacle_cost_scale * (0.5 + 3.0 * ratio * ratio)  # Reduced from 1.0 + 8.0
        return min(cost, 10.0 * self._obstacle_cost_scale)  # Hard cap to prevent runaway costs

    def _nearest_obstacle_dist(self, x: float, y: float) -> float:
        if self._lidar_xy is None or len(self._lidar_xy) == 0:
            return float("inf")
        dx = self._lidar_xy[:, 0] - x
        dy = self._lidar_xy[:, 1] - y
        return float(np.min(np.hypot(dx, dy)))

    def _should_publish_now(self, pose_x: float, pose_y: float, goal_x: float, goal_y: float) -> bool:
        now = self.get_clock().now()
        if self._last_publish_time is None:
            return True

        nearest_obs = self._nearest_obstacle_dist(pose_x, pose_y)
        target_period = self._near_publish_period_sec if nearest_obs < self._near_obstacle_dist_m else self._far_publish_period_sec
        elapsed = (now - self._last_publish_time).nanoseconds * 1e-9
        if elapsed >= target_period:
            return True

        cur_pose_xy = np.asarray([pose_x, pose_y], dtype=float)
        cur_goal_xy = np.asarray([goal_x, goal_y], dtype=float)
        moved = 0.0 if self._last_publish_pose_xy is None else float(np.linalg.norm(cur_pose_xy - self._last_publish_pose_xy))
        goal_shift = 0.0 if self._last_goal_xy is None else float(np.linalg.norm(cur_goal_xy - self._last_goal_xy))

        # Force quick graph refresh if goal jumps or robot makes a meaningful move.
        return moved > self._far_motion_threshold_m or goal_shift > 0.20

    def _publish_graph(self) -> None:
        if self._pose is None:
            return

        pose = self._pose.pose.position
        frame_id = self._graph_frame_id if self._graph_frame_id else (self._pose.header.frame_id or "map")

        if self._goal is None:
            gx = pose.x + 3.0
            gy = pose.y
            gz = pose.z
        else:
            gx = self._goal.pose.position.x
            gy = self._goal.pose.position.y
            gz = self._goal.pose.position.z

        dx = gx - pose.x
        dy = gy - pose.y
        dist = math.hypot(dx, dy)

        if not self._should_publish_now(pose.x, pose.y, gx, gy):
            return

        if dist < 1e-3:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / dist, dy / dist
        nx, ny = -uy, ux

        n_segments = int(max(2, min(self._max_segments, math.ceil(max(dist, 1.0) / self._segment_spacing) + 1)))

        nodes: list[Node] = []
        edges: list[Edge] = []

        center_indices: list[int] = []
        left_indices: list[int] = []
        right_indices: list[int] = []
        node_costs: list[float] = []

        z = pose.z
        for i in range(n_segments):
            t = i / max(n_segments - 1, 1)
            cx = pose.x + t * (gx - pose.x)
            cy = pose.y + t * (gy - pose.y)

            c_idx = len(nodes)
            nodes.append(self._make_node(cx, cy, z, frame_id, frontier_bias=1.0))
            center_indices.append(c_idx)
            node_costs.append(self._node_obstacle_cost(cx, cy))

            if self._lateral_offset > 0.0:
                lx = cx + nx * self._lateral_offset
                ly = cy + ny * self._lateral_offset
                rx = cx - nx * self._lateral_offset
                ry = cy - ny * self._lateral_offset

                l_idx = len(nodes)
                nodes.append(self._make_node(lx, ly, z, frame_id, frontier_bias=0.8))
                left_indices.append(l_idx)
                node_costs.append(self._node_obstacle_cost(lx, ly))

                r_idx = len(nodes)
                nodes.append(self._make_node(rx, ry, z, frame_id, frontier_bias=0.8))
                right_indices.append(r_idx)
                node_costs.append(self._node_obstacle_cost(rx, ry))

                edges.append(self._make_edge(c_idx, l_idx, self._lateral_offset * (1.0 + 0.5 * (node_costs[c_idx] + node_costs[l_idx]))))
                edges.append(self._make_edge(c_idx, r_idx, self._lateral_offset * (1.0 + 0.5 * (node_costs[c_idx] + node_costs[r_idx]))))

        for i in range(len(center_indices) - 1):
            a = center_indices[i]
            b = center_indices[i + 1]
            ax, ay = nodes[a].pose.position.x, nodes[a].pose.position.y
            bx, by = nodes[b].pose.position.x, nodes[b].pose.position.y
            base = math.hypot(bx - ax, by - ay)
            edges.append(self._make_edge(a, b, base * (1.0 + 0.5 * (node_costs[a] + node_costs[b]))))

        for branch in (left_indices, right_indices):
            for i in range(len(branch) - 1):
                a = branch[i]
                b = branch[i + 1]
                ax, ay = nodes[a].pose.position.x, nodes[a].pose.position.y
                bx, by = nodes[b].pose.position.x, nodes[b].pose.position.y
                base = math.hypot(bx - ax, by - ay)
                edges.append(self._make_edge(a, b, base * (1.0 + 0.5 * (node_costs[a] + node_costs[b]))))

        # Diagnostic logging
        max_cost = max(node_costs) if node_costs else 0.0
        max_edge_cost = max([e.traversability[0].traversability_cost for e in edges]) if edges else 0.0
        lidar_count = len(self._lidar_xy) if self._lidar_xy is not None else 0
        
        if lidar_count > 0 or max_cost > 0.1:
            self.get_logger().info(
                f"Graph: {len(nodes)} nodes, {len(edges)} edges | "
                f"LiDAR points: {lidar_count} | "
                f"Max node cost: {max_cost:.2f} | Max edge cost: {max_edge_cost:.2f}"
            )

        graph = NavigationGraph()
        graph.header.stamp = self.get_clock().now().to_msg()
        graph.header.frame_id = frame_id
        graph.trav_classes = ["default"]
        graph.nodes = nodes
        graph.edges = edges
        graph.current_node_idx = 0

        self._graph_pub.publish(graph)
        self._last_publish_time = self.get_clock().now()
        self._last_publish_pose_xy = np.asarray([pose.x, pose.y], dtype=float)
        self._last_goal_xy = np.asarray([gx, gy], dtype=float)


def main() -> None:
    rclpy.init()
    node = MockNavGraphPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
