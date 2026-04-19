"""
mock_nav_graph_publisher.py

Publishes a mock/dummy NavigationGraph for development and testing.
This is useful when the external visual_navigation stack is not available.

Generates a simple grid of waypoints in a rectangular pattern.
"""

import rclpy
from rclpy.node import Node

from graphnav_msgs.msg import NavigationGraph, Node as GraphNode, Edge
from geometry_msgs.msg import Point
from std_msgs.msg import Header


class MockNavGraphPublisher(Node):

    def __init__(self):
        super().__init__('mock_nav_graph_publisher')

        self.declare_parameter('graph_frame', 'map')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('grid_spacing', 2.0)
        self.declare_parameter('grid_width', 5)
        self.declare_parameter('grid_height', 5)

        frame = self.get_parameter('graph_frame').value
        rate = self.get_parameter('publish_rate_hz').value
        spacing = self.get_parameter('grid_spacing').value
        width = int(self.get_parameter('grid_width').value)
        height = int(self.get_parameter('grid_height').value)

        self._pub = self.create_publisher(NavigationGraph, 'scored_nav_graph', 10)
        self._timer = self.create_timer(1.0 / rate, self._publish_graph)

        self._frame = frame
        self._spacing = spacing
        self._width = width
        self._height = height
        self._count = 0

        self.get_logger().info(
            f'Mock NavigationGraph: {width}×{height} grid '
            f'(spacing={spacing}m) on frame "{frame}"'
        )

    def _create_grid_graph(self):
        """Create a grid of waypoints."""
        graph = NavigationGraph()
        graph.header = Header(frame_id=self._frame, stamp=self.get_clock().now().to_msg())

        nodes = []
        edges = []

        # Create grid nodes
        for row in range(self._height):
            for col in range(self._width):
                node_id = row * self._width + col
                x = col * self._spacing
                y = row * self._spacing
                z = 0.0

                n = GraphNode()
                n.node_id = node_id
                n.position = Point(x=x, y=y, z=z)
                n.traversability_score = 0.9  # Good traversability
                nodes.append(n)

        # Create edges (grid connectivity)
        for row in range(self._height):
            for col in range(self._width):
                node_id = row * self._width + col

                # Right neighbor
                if col < self._width - 1:
                    neighbor_id = row * self._width + (col + 1)
                    e = Edge()
                    e.start_node_id = node_id
                    e.end_node_id = neighbor_id
                    e.traversability_score = 0.9
                    edges.append(e)

                # Bottom neighbor
                if row < self._height - 1:
                    neighbor_id = (row + 1) * self._width + col
                    e = Edge()
                    e.start_node_id = node_id
                    e.end_node_id = neighbor_id
                    e.traversability_score = 0.9
                    edges.append(e)

        graph.nodes = nodes
        graph.edges = edges
        return graph

    def _publish_graph(self):
        graph = self._create_grid_graph()
        self._pub.publish(graph)
        self._count += 1

        if self._count % 10 == 0:  # Log every 10th publish
            self.get_logger().debug(
                f'Published NavigationGraph: {len(graph.nodes)} nodes, {len(graph.edges)} edges'
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
