"""
scenario_manager_node.py
Loads a YAML scenario on startup and manages runtime obstacle spawning in Gazebo.

Parameters
----------
scenario_file : str
    Absolute path to a scenario YAML file (or scenario name in share/sim_scenarios/scenarios/).
    Empty string = start with no obstacles.
models_path : str
    Directory containing Gazebo SDF obstacle models (model.sdf files).
    Defaults to share/sim_worlds/models.

Services exposed
----------------
/sim/clear_obstacles   std_srvs/Trigger  — Remove all obstacles spawned by this node.
/sim/load_scenario     std_srvs/Trigger  — Reload (or load for the first time) the
                                           currently configured scenario_file parameter.

Gazebo services used
--------------------
/spawn_entity   gazebo_msgs/srv/SpawnEntity
/delete_entity  gazebo_msgs/srv/DeleteEntity

Scenario YAML format
--------------------
  name: my_scenario
  obstacles:
    - name: box_1
      model: obstacle_box
      pose: {x: 2.0, y: 1.0, z: 0.5, roll: 0.0, pitch: 0.0, yaw: 0.0}
    - name: cylinder_1
      model: obstacle_cylinder
      pose: {x: -1.5, y: 2.0, z: 0.5}   # roll/pitch/yaw default to 0

Ground z values:
  obstacle_box      → z = 0.50  (half of 1.0 m height)
  obstacle_cylinder → z = 0.50
  obstacle_wall     → z = 0.40  (half of 0.8 m height)
  obstacle_pallet   → z = 0.07  (half of 0.14 m height)
"""

import os
import yaml

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from std_srvs.srv import Trigger
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose

from ament_index_python.packages import get_package_share_directory


class ScenarioManager(Node):

    def __init__(self):
        super().__init__('scenario_manager')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('scenario_file', '')
        self.declare_parameter(
            'models_path',
            os.path.join(get_package_share_directory('sim_worlds'), 'models'),
        )

        self._scenario_file = self.get_parameter('scenario_file').value
        self._models_path   = self.get_parameter('models_path').value

        # ── Gazebo clients ─────────────────────────────────────────────
        cb = ReentrantCallbackGroup()
        self._spawn_client  = self.create_client(SpawnEntity,  '/spawn_entity',  callback_group=cb)
        self._delete_client = self.create_client(DeleteEntity, '/delete_entity', callback_group=cb)

        # ── Exposed services ───────────────────────────────────────────
        self.create_service(Trigger, '/sim/clear_obstacles', self._clear_cb)
        self.create_service(Trigger, '/sim/load_scenario',  self._load_cb)

        # ── State: names of obstacles currently spawned by this node ──
        self._spawned: list[str] = []

        # ── Auto-load on startup ───────────────────────────────────────
        if self._scenario_file:
            self.create_timer(2.0, self._startup_load)  # wait for Gazebo to start
        else:
            self.get_logger().info('scenario_manager ready (no scenario_file configured)')

    # ──────────────────────────────────────────────────────────────────
    # Startup load (runs once, then cancels itself)
    # ──────────────────────────────────────────────────────────────────

    def _startup_load(self):
        # Cancel the one-shot timer immediately
        self._startup_timer.cancel()
        self._load_scenario(self._scenario_file)

    # Stash timer reference so we can cancel it
    def create_timer(self, period, callback):
        t = super().create_timer(period, callback)
        if callback == self._startup_load:
            self._startup_timer = t
        return t

    # ──────────────────────────────────────────────────────────────────
    # Service handlers
    # ──────────────────────────────────────────────────────────────────

    def _clear_cb(self, _req, response):
        errors = self._clear_all()
        if errors:
            response.success = False
            response.message = f'Failed to delete: {errors}'
        else:
            response.success = True
            response.message = 'All obstacles cleared'
        return response

    def _load_cb(self, _req, response):
        scenario_file = self.get_parameter('scenario_file').value
        if not scenario_file:
            response.success = False
            response.message = 'scenario_file parameter is empty'
            return response

        self._clear_all()
        ok = self._load_scenario(scenario_file)
        response.success = ok
        response.message = 'Scenario loaded' if ok else 'Failed to load scenario (check logs)'
        return response

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _resolve_scenario_path(self, path: str) -> str:
        """
        Accept either:
          - absolute path                    → use as-is
          - relative basename (no dir sep)   → look up in share/sim_scenarios/scenarios/
        """
        if os.path.isabs(path):
            return path
        # Strip extension if caller omitted it
        name = path if path.endswith('.yaml') else path + '.yaml'
        share = get_package_share_directory('sim_scenarios')
        return os.path.join(share, 'scenarios', name)

    def _load_scenario(self, path: str) -> bool:
        resolved = self._resolve_scenario_path(path)
        if not os.path.isfile(resolved):
            self.get_logger().error(f'Scenario file not found: {resolved}')
            return False

        with open(resolved, 'r') as f:
            data = yaml.safe_load(f)

        obstacles = data.get('obstacles', [])
        self.get_logger().info(
            f"Loading scenario '{data.get('name', resolved)}' "
            f"({len(obstacles)} obstacles)"
        )

        success = True
        for obs in obstacles:
            ok = self._spawn_obstacle(
                name=obs['name'],
                model_name=obs['model'],
                pose_dict=obs.get('pose', {}),
            )
            if not ok:
                success = False

        return success

    def _spawn_obstacle(self, name: str, model_name: str, pose_dict: dict) -> bool:
        # Load the SDF from models_path
        sdf_path = os.path.join(self._models_path, model_name, 'model.sdf')
        if not os.path.isfile(sdf_path):
            self.get_logger().error(f'Model SDF not found: {sdf_path}')
            return False

        with open(sdf_path, 'r') as f:
            entity_xml = f.read()

        # Build pose
        pose = Pose()
        pose.position.x = float(pose_dict.get('x', 0.0))
        pose.position.y = float(pose_dict.get('y', 0.0))
        pose.position.z = float(pose_dict.get('z', 0.0))

        roll  = float(pose_dict.get('roll',  0.0))
        pitch = float(pose_dict.get('pitch', 0.0))
        yaw   = float(pose_dict.get('yaw',   0.0))
        pose.orientation = _euler_to_quat(roll, pitch, yaw)

        # Wait for service
        if not self._spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/spawn_entity service not available')
            return False

        req = SpawnEntity.Request()
        req.name          = name
        req.xml           = entity_xml
        req.initial_pose  = pose
        req.reference_frame = 'world'

        future = self._spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            self.get_logger().error(f'SpawnEntity call failed for {name}')
            return False
        if not future.result().success:
            self.get_logger().error(
                f'SpawnEntity rejected {name}: {future.result().status_message}'
            )
            return False

        self.get_logger().info(f'Spawned obstacle: {name} ({model_name})')
        self._spawned.append(name)
        return True

    def _delete_obstacle(self, name: str) -> bool:
        if not self._delete_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/delete_entity service not available')
            return False

        req = DeleteEntity.Request()
        req.name = name

        future = self._delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None or not future.result().success:
            msg = future.result().status_message if future.result() else 'timeout'
            self.get_logger().warning(f'DeleteEntity failed for {name}: {msg}')
            return False

        self.get_logger().info(f'Deleted obstacle: {name}')
        return True

    def _clear_all(self) -> list[str]:
        errors = []
        for name in list(self._spawned):
            ok = self._delete_obstacle(name)
            if ok:
                self._spawned.remove(name)
            else:
                errors.append(name)
        return errors


# ──────────────────────────────────────────────────────────────────────
# Utility: Euler (RPY) → quaternion  (no external dependencies)
# ──────────────────────────────────────────────────────────────────────

import math
from geometry_msgs.msg import Quaternion


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


# ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ScenarioManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
