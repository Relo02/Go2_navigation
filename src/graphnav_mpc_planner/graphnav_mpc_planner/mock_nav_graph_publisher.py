"""
mock_nav_graph_publisher.py

Publishes an obstacle-aware NavigationGraph for simulation without wildos.

Grid of waypoints in the map frame.  Edges that pass through obstacle
regions (detected from /lidar/points_filtered) are removed so the
Dijkstra planner naturally routes around walls.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from graphnav_msgs.msg import NavigationGraph, Node as GraphNode, Edge
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class MockNavGraphPublisher(Node):

    def __init__(self):
        super().__init__('mock_nav_graph_publisher')

        self.declare_parameter('graph_frame',         'map')
        self.declare_parameter('publish_rate_hz',      1.0)
        self.declare_parameter('grid_spacing',         2.0)
        self.declare_parameter('grid_width',           5)
        self.declare_parameter('grid_height',          5)
        # Obstacle inflation: edge is blocked when any lidar point is
        # within this distance of the edge midpoint.
        self.declare_parameter('obs_block_radius',     0.8)

        self._frame         = self.get_parameter('graph_frame').value
        rate                = self.get_parameter('publish_rate_hz').value
        self._spacing       = self.get_parameter('grid_spacing').value
        self._width         = int(self.get_parameter('grid_width').value)
        self._height        = int(self.get_parameter('grid_height').value)
        self._obs_r         = float(self.get_parameter('obs_block_radius').value)

        # Latest obstacle point cloud (xy, world frame)
        self._obs_xy: np.ndarray | None = None
        # Robot position (world frame) for current_node_idx
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            PointCloud2, '/lidar/points_filtered', self._cloud_cb, sensor_qos
        )
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, sensor_qos
        )

        self._pub   = self.create_publisher(NavigationGraph, '/scored_nav_graph', 10)
        self._timer = self.create_timer(1.0 / rate, self._publish_graph)
        self._count = 0

        self.get_logger().info(
            f'MockNavGraph: {self._width}×{self._height} grid '
            f'spacing={self._spacing}m obs_block_radius={self._obs_r}m'
        )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _cloud_cb(self, msg: PointCloud2) -> None:
        try:
            pts = list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
            if pts:
                arr = np.array([(p[0], p[1]) for p in pts], dtype=float)
                finite = np.isfinite(arr).all(axis=1)
                self._obs_xy = arr[finite] if np.any(finite) else None
            else:
                self._obs_xy = None
        except Exception as e:
            self.get_logger().warning(f'Cloud parse error: {e}', throttle_duration_sec=5.0)

    # ── Graph building ─────────────────────────────────────────────────

    def _edge_blocked(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True if any obstacle point is within obs_block_radius of any sample along the edge."""
        if self._obs_xy is None or len(self._obs_xy) == 0:
            return False
        # Sample at 5 points along the edge to catch obstacles near endpoints too
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            mx = x0 + t * (x1 - x0)
            my = y0 + t * (y1 - y0)
            dists = np.hypot(self._obs_xy[:, 0] - mx, self._obs_xy[:, 1] - my)
            if bool(np.any(dists < self._obs_r)):
                return True
        return False

    def _create_grid_graph(self) -> NavigationGraph:
        graph = NavigationGraph()
        graph.header = Header(frame_id=self._frame, stamp=self.get_clock().now().to_msg())

        nodes = []
        nearest_idx  = 0
        nearest_dist = float('inf')
        for row in range(self._height):
            for col in range(self._width):
                nx = float(col * self._spacing)
                ny = float(row * self._spacing)
                d  = math.hypot(nx - self._robot_x, ny - self._robot_y)
                idx = row * self._width + col
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx  = idx
                n = GraphNode()
                n.pose = Pose()
                n.pose.position.x = nx
                n.pose.position.y = ny
                n.pose.position.z = 0.0
                n.pose.orientation.w = 1.0
                nodes.append(n)

        graph.current_node_idx = nearest_idx

        edges = []
        blocked = 0
        # 8-connected grid: cardinal + diagonal neighbors
        neighbors = [
            (1, 0),   # right
            (0, 1),   # up
            (1, 1),   # up-right diagonal
            (-1, 1),  # up-left diagonal
        ]
        for row in range(self._height):
            for col in range(self._width):
                idx = row * self._width + col
                x0  = float(col * self._spacing)
                y0  = float(row * self._spacing)

                for dc, dr in neighbors:
                    nc, nr = col + dc, row + dr
                    if not (0 <= nc < self._width and 0 <= nr < self._height):
                        continue
                    x1 = float(nc * self._spacing)
                    y1 = float(nr * self._spacing)
                    if not self._edge_blocked(x0, y0, x1, y1):
                        e = Edge()
                        e.from_idx = idx
                        e.to_idx   = nr * self._width + nc
                        edges.append(e)
                    else:
                        blocked += 1

        graph.nodes = nodes
        graph.edges = edges

        if blocked > 0:
            self.get_logger().debug(
                f'Graph: {len(nodes)} nodes, {len(edges)} edges ({blocked} blocked by obstacles)',
                throttle_duration_sec=2.0,
            )
        return graph

    def _publish_graph(self) -> None:
        graph = self._create_grid_graph()
        self._pub.publish(graph)
        self._count += 1

        if self._count == 1:
            self.get_logger().info(
                f'Publishing NavigationGraph to /scored_nav_graph '
                f'({len(graph.nodes)} nodes, {len(graph.edges)} edges)'
            )


def main(args=None):
    rclpy.init(args=args)
    node = MockNavGraphPublisher()
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
