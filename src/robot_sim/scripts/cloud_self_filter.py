#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class CloudSelfFilter(Node):
    """Mask the simulated robot body out of the voxel-layer cloud feed."""

    def __init__(self) -> None:
        super().__init__("cloud_self_filter")

        self.declare_parameter("input_topic", "/lidar/points")
        self.declare_parameter("output_topic", "/lidar/points_filtered")
        self.declare_parameter("lidar_x", 0.25)
        self.declare_parameter("lidar_y", -0.038)
        self.declare_parameter("lidar_z", -0.03)
        self.declare_parameter("base_z", 0.335)
        self.declare_parameter("min_world_z", 0.06)
        self.declare_parameter("robot_min_x", -0.40)
        self.declare_parameter("robot_max_x", 0.38)
        self.declare_parameter("robot_min_y", -0.26)
        self.declare_parameter("robot_max_y", 0.26)
        self.declare_parameter("robot_min_z", -0.42)
        self.declare_parameter("robot_max_z", 0.18)
        self.declare_parameter("min_radius", 0.25)
        self.declare_parameter("lidar_roll", 0.0)
        self.declare_parameter("lidar_pitch", 0.0)
        self.declare_parameter("lidar_yaw", 0.0)
        self.declare_parameter("min_ray_elevation_deg", 0.0)
        self.declare_parameter("publish_in_world_frame", True)
        self.declare_parameter("world_frame_id", "map")
        self.declare_parameter("pose_topic", "/go2/pose")

        self._lidar_x = float(self.get_parameter("lidar_x").value)
        self._lidar_y = float(self.get_parameter("lidar_y").value)
        self._lidar_z = float(self.get_parameter("lidar_z").value)
        self._base_z = float(self.get_parameter("base_z").value)
        self._lidar_roll = float(self.get_parameter("lidar_roll").value)
        self._lidar_pitch = float(self.get_parameter("lidar_pitch").value)
        self._lidar_yaw = float(self.get_parameter("lidar_yaw").value)
        self._min_ray_elevation_deg = float(
            self.get_parameter("min_ray_elevation_deg").value
        )
        self._publish_in_world_frame = bool(
            self.get_parameter("publish_in_world_frame").value
        )
        self._world_frame_id = str(self.get_parameter("world_frame_id").value)
        self._min_world_z = float(self.get_parameter("min_world_z").value)
        self._robot_min_x = float(self.get_parameter("robot_min_x").value)
        self._robot_max_x = float(self.get_parameter("robot_max_x").value)
        self._robot_min_y = float(self.get_parameter("robot_min_y").value)
        self._robot_max_y = float(self.get_parameter("robot_max_y").value)
        self._robot_min_z = float(self.get_parameter("robot_min_z").value)
        self._robot_max_z = float(self.get_parameter("robot_max_z").value)
        self._min_radius = float(self.get_parameter("min_radius").value)

        cr = math.cos(self._lidar_roll)
        sr = math.sin(self._lidar_roll)
        cp = math.cos(self._lidar_pitch)
        sp = math.sin(self._lidar_pitch)
        cy = math.cos(self._lidar_yaw)
        sy = math.sin(self._lidar_yaw)
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        self._r00 = cy * cp
        self._r01 = cy * sp * sr - sy * cr
        self._r02 = cy * sp * cr + sy * sr
        self._r10 = sy * cp
        self._r11 = sy * sp * sr + cy * cr
        self._r12 = sy * sp * cr - cy * sr
        self._r20 = -sp
        self._r21 = cp * sr
        self._r22 = cp * cr

        self._pose: PoseStamped | None = None
        self._qw = 1.0
        self._qx = 0.0
        self._qy = 0.0
        self._qz = 0.0
        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)

        self._publisher = self.create_publisher(
            PointCloud2, output_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2, input_topic, self._cloud_cb, qos_profile_sensor_data
        )
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 10)
        # Debug state
        self._pose_received = False
        self._cloud_count = 0
        self._filtered_count = 0

        self.get_logger().info(f'CloudSelfFilter initialized')
        self.get_logger().info(f'  Input:  {input_topic}')
        self.get_logger().info(f'  Output: {output_topic} (world frame: {self._world_frame_id})')
        self.get_logger().info(f'  Pose:   {pose_topic}')
    def _pose_cb(self, msg: PoseStamped) -> None:
        if not self._pose_received:
            self.get_logger().info(f'✓ Received pose from frame "{msg.header.frame_id}"')
            self._pose_received = True
        self._pose = msg
        self._px = float(msg.pose.position.x)
        self._py = float(msg.pose.position.y)
        self._pz = float(msg.pose.position.z)
        self._qx = float(msg.pose.orientation.x)
        self._qy = float(msg.pose.orientation.y)
        self._qz = float(msg.pose.orientation.z)
        self._qw = float(msg.pose.orientation.w)

    def _base_to_world(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Rotate+translate a base-frame point into world frame using /go2/pose."""
        # Quaternion rotation matrix (normalized quaternion assumed).
        qx, qy, qz, qw = self._qx, self._qy, self._qz, self._qw
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)
        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)
        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        wx = self._px + (r00 * x + r01 * y + r02 * z)
        wy = self._py + (r10 * x + r11 * y + r12 * z)
        wz = self._pz + (r20 * x + r21 * y + r22 * z)
        return wx, wy, wz

    def _lidar_to_base(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Rotate a point from lidar frame into base frame using roll/pitch/yaw."""
        x_b = self._r00 * x + self._r01 * y + self._r02 * z
        y_b = self._r10 * x + self._r11 * y + self._r12 * z
        z_b = self._r20 * x + self._r21 * y + self._r22 * z
        return x_b, y_b, z_b

    def _inside_self_box(self, x: float, y: float, z: float, frame_id: str) -> bool:
        frame = frame_id.lower()
        if frame in ("map", "odom", "world"):
            # Self-box masking is defined in base/lidar coordinates.
            # For world-frame clouds rely on min_radius + ground removal instead.
            return False

        x_rot, y_rot, z_rot = self._lidar_to_base(x, y, z)
        x_base = x_rot + self._lidar_x
        y_base = y_rot + self._lidar_y
        z_base = z_rot + self._lidar_z

        return (
            self._robot_min_x <= x_base <= self._robot_max_x
            and self._robot_min_y <= y_base <= self._robot_max_y
            and self._robot_min_z <= z_base <= self._robot_max_z
        )

    def _below_ground_gate(self, x: float, y: float, z: float, frame_id: str) -> bool:
        # If the cloud is already in a world frame, z is absolute.
        frame = frame_id.lower()
        if frame in ("map", "odom", "world"):
            return z < self._min_world_z

        # Otherwise assume lidar frame: convert to base frame and estimate world Z.
        _, _, z_rot = self._lidar_to_base(x, y, z)
        z_in_base = self._lidar_z + z_rot
        return (self._base_z + z_in_base) < self._min_world_z

    def _below_elevation_gate(self, x: float, y: float, z: float, frame_id: str) -> bool:
        # Elevation gate is only valid for lidar-frame clouds.
        frame = frame_id.lower()
        if frame in ("map", "odom", "world"):
            return False

        # Reject steeply downward beams in base frame (e.g., ground returns).
        x_b, y_b, z_b = self._lidar_to_base(x, y, z)
        xy_norm = math.hypot(x_b, y_b)
        if xy_norm <= 1e-6:
            return False
        elev_deg = math.degrees(math.atan2(z_b, xy_norm))
        return elev_deg < (-self._min_ray_elevation_deg)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        self._cloud_count += 1
        filtered_points = []
        frame_id = msg.header.frame_id
        frame = frame_id.lower()

        use_pose_transform = (
            self._publish_in_world_frame
            and frame not in ("map", "odom", "world")
        )

        # Log first cloud message
        if self._cloud_count == 1:
            num_points = len(list(point_cloud2.read_points(msg)))
            self.get_logger().info(
                f'✓ Received first point cloud: {num_points} points from frame "{frame_id}"'
            )

        for point in point_cloud2.read_points(msg, field_names=None, skip_nans=False):
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])

            # Drop NaN/inf points — never pass them downstream
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue

            if math.sqrt(x * x + y * y + z * z) < self._min_radius:
                continue
            if self._below_elevation_gate(x, y, z, frame_id):
                continue
            if self._inside_self_box(x, y, z, frame_id):
                continue

            if use_pose_transform:
                # Transform lidar-frame point -> base -> world.
                if self._pose is None:
                    # Drop until pose is available; avoids feeding wrong-frame points downstream.
                    if self._cloud_count == 1:
                        self.get_logger().warning(
                            'Cloud received but pose not yet available — waiting for /go2/pose'
                        )
                    continue
                xb, yb, zb = self._lidar_to_base(x, y, z)
                xb += self._lidar_x
                yb += self._lidar_y
                zb += self._lidar_z
                xw, yw, zw = self._base_to_world(xb, yb, zb)
                if zw < self._min_world_z:
                    continue
                out = list(point)
                out[0] = xw
                out[1] = yw
                out[2] = zw
                filtered_points.append(tuple(out))
                continue

            if self._below_ground_gate(x, y, z, frame_id):
                continue

            filtered_points.append(tuple(point))

        out_header = msg.header
        if use_pose_transform:
            if self._pose is None:
                # No pose yet — drop entire scan rather than publishing sensor-frame points
                # with a world-frame label (or no label at all).
                return
            out_header.frame_id = self._world_frame_id
        filtered = point_cloud2.create_cloud(out_header, msg.fields, filtered_points)
        filtered.is_dense = msg.is_dense
        self._publisher.publish(filtered)        
        # Throttled debug: confirm filtered clouds are publishing
        self._filtered_count += 1
        if self._filtered_count == 1:
            self.get_logger().info(
                f'✓ Publishing filtered cloud to {self._pub.topic_name}: {len(filtered_points)} points'
            )
        elif self._filtered_count % 20 == 0:
            self.get_logger().debug(
                f'Filtering: in={len(list(point_cloud2.read_points(msg)))} out={len(filtered_points)}'
            )

def main() -> None:
    rclpy.init()
    node = CloudSelfFilter()
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
