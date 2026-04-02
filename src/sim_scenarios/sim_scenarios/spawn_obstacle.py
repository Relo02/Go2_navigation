"""
spawn_obstacle.py
One-shot CLI tool for runtime obstacle manipulation in a running Gazebo simulation.

Usage
-----
Spawn:
  ros2 run sim_scenarios spawn_obstacle \\
      --name box_1 --model obstacle_box --x 2.0 --y 1.0 --z 0.5

Delete:
  ros2 run sim_scenarios spawn_obstacle --delete box_1

Move (delete + re-spawn at new pose):
  ros2 run sim_scenarios spawn_obstacle \\
      --move box_1 --model obstacle_box --x 3.0 --y 2.0 --z 0.5

Default z values that place models flush on the ground plane:
  obstacle_box      → --z 0.50
  obstacle_cylinder → --z 0.50
  obstacle_wall     → --z 0.40
  obstacle_pallet   → --z 0.07
"""

import argparse
import math
import os
import sys

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose, Quaternion

from ament_index_python.packages import get_package_share_directory


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw   * 0.5)
    sy = math.sin(yaw   * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def spawn(node: Node, args, models_path: str) -> bool:
    sdf_path = os.path.join(models_path, args.model, 'model.sdf')
    if not os.path.isfile(sdf_path):
        node.get_logger().error(f'Model SDF not found: {sdf_path}')
        return False

    with open(sdf_path, 'r') as f:
        xml = f.read()

    pose = Pose()
    pose.position.x = args.x
    pose.position.y = args.y
    pose.position.z = args.z
    pose.orientation = _euler_to_quat(args.roll, args.pitch, args.yaw)

    client = node.create_client(SpawnEntity, '/spawn_entity')
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error('/spawn_entity service not available')
        return False

    req = SpawnEntity.Request()
    req.name           = args.name
    req.xml            = xml
    req.initial_pose   = pose
    req.reference_frame = 'world'

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

    if future.result() is None or not future.result().success:
        msg = future.result().status_message if future.result() else 'timeout'
        node.get_logger().error(f'SpawnEntity failed: {msg}')
        return False

    node.get_logger().info(f'Spawned {args.name} ({args.model}) at ({args.x},{args.y},{args.z})')
    return True


def delete(node: Node, name: str) -> bool:
    client = node.create_client(DeleteEntity, '/delete_entity')
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error('/delete_entity service not available')
        return False

    req = DeleteEntity.Request()
    req.name = name

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

    if future.result() is None or not future.result().success:
        msg = future.result().status_message if future.result() else 'timeout'
        node.get_logger().error(f'DeleteEntity failed: {msg}')
        return False

    node.get_logger().info(f'Deleted {name}')
    return True


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Spawn, delete, or move a Gazebo obstacle.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # --delete and --move are optional; omitting both means spawn (default action).
    action = parser.add_mutually_exclusive_group(required=False)
    action.add_argument('--delete', metavar='NAME',
                        help='Delete an existing obstacle by name')
    action.add_argument('--move', metavar='NAME',
                        help='Reposition an existing obstacle (delete + re-spawn)')

    parser.add_argument('--name',  help='Obstacle name (for spawn)')
    parser.add_argument('--model', help='Model name (obstacle_box, obstacle_cylinder, …)')
    parser.add_argument('--x',     type=float, default=0.0)
    parser.add_argument('--y',     type=float, default=0.0)
    parser.add_argument('--z',     type=float, default=0.5)
    parser.add_argument('--roll',  type=float, default=0.0)
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--yaw',   type=float, default=0.0)
    parser.add_argument('--models-path', default='',
                        help='Override path to model SDF directory')

    # Split ROS args from program args
    rclpy.init(args=args)
    parsed = parser.parse_args(sys.argv[1:])

    node = Node('spawn_obstacle_cli')

    models_path = parsed.models_path or os.path.join(
        get_package_share_directory('sim_worlds'), 'models'
    )

    ok = True
    try:
        if parsed.delete:
            ok = delete(node, parsed.delete)

        elif parsed.move:
            if not parsed.model:
                parser.error('--move requires --model')
            parsed.name = parsed.move
            ok = delete(node, parsed.move)
            if ok:
                ok = spawn(node, parsed, models_path)

        else:
            # Default action: spawn. Requires --name and --model.
            if not parsed.name:
                parser.error('spawn requires --name')
            if not parsed.model:
                parser.error('spawn requires --model')
            ok = spawn(node, parsed, models_path)

    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if ok else 1)
