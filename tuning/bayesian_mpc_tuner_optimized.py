#!/usr/bin/env python3

"""
Bayesian MPC Tuner for Go2 — hardened for long overnight runs.

Per-trial artifacts saved to tuning_results/trial_NNN/:
  planner_params.yaml         — exact YAML used (parameter history)
  scenario_<name>/rosbag/     — ros2 bag record for every scenario
  gp_surrogate.json           — ARD-GP kernel params fit on accumulated data
  tpe_state.json              — hyperopt TPE Trials state snapshot
  metadata.json               — params, per-scenario scores, timing

Root-level artifacts:
  best_planner_params.yaml    — YAML for the best trial so far
  results.json                — all trial summaries + best
  gp_history.json             — GP kernel param evolution (length scales, noise)
  convergence.png             — score vs trial
  param_importance.png        — GP-derived parameter sensitivity
  length_scales.png           — GP length scale evolution

Hardening changes applied (see FIX-N labels throughout):
  FIX-1  ROS 2 lifecycle: rclpy.init() once at program start, nodes destroyed per scenario
  FIX-2  Process management: PID tracking, SIGTERM→wait→SIGKILL, no pkill -9
  FIX-3  Simulation reuse: Gazebo kept alive across scenarios, reset between them
  FIX-4  Early termination: abort scenario on stall / no-progress / MPC failure rate
  FIX-5  PointCloud2 parsing: sensor_msgs_py with safe fallback
  FIX-6  Score function: MPC penalty, NaN guard, clamping, better obs weight
  FIX-7  Disk usage: topic filtering, bag compression, keep-last-N cleanup policy
  FIX-8  GP surrogate: min 10 samples, defensive error handling, smoothed importance
  FIX-9  Logging: structured with timestamps/IDs, failure reasons, heartbeat
  FIX-10 Runtime: condition-based waits, configurable timeouts, reduced idle sleep
  FIX-11 Safety guards: global trial timeout, catch-all, clean recovery
  FIX-12 Code quality: type hints, modularity, removed duplication
"""

import copy
import json
import logging
import os
import signal
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

# FIX-1: rclpy imported at top level; init/shutdown managed by the tuner lifecycle
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as NavPath
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# ─── Structured logging setup (FIX-9) ────────────────────────────────────────

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mpc_tuner")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)-7s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / "tuner.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Module-level logger (reconfigured in BayesianMPCTuner.__init__)
log = logging.getLogger("mpc_tuner")


# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent.parent.resolve()
BASE_PARAMS  = REPO_ROOT / "src/a_star_mpc_planner/config/planner_params.yaml"
RESULTS_DIR  = Path(os.environ.get("TUNING_RESULTS_DIR",
                    "/media/lorenzo/writable/tuning_results"))
ROS_SETUP    = "/opt/ros/humble/setup.bash"
PKG_SETUP    = REPO_ROOT / "install/setup.bash"

# ─── Search space ─────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "mpc_Q_x":              hp.uniform("mpc_Q_x",              50.0,  500.0),
    "mpc_Q_y":              hp.uniform("mpc_Q_y",              50.0,  500.0),
    "mpc_Q_yaw":            hp.uniform("mpc_Q_yaw",             0.1,   15.0),
    "mpc_Q_terminal":       hp.uniform("mpc_Q_terminal",       20.0,  300.0),
    "mpc_W_obs_sigmoid":    hp.uniform("mpc_W_obs_sigmoid",    50.0,  400.0),
    "grid_std":             hp.uniform("grid_std",              0.1,   0.25),
}

PARAM_NAMES = list(SEARCH_SPACE.keys())

# ─── Trial settings ───────────────────────────────────────────────────────────

MAX_EVALS             = 30
N_RANDOM_INIT         = 8
SCENARIO_TIMEOUT      = 120     # default per-scenario timeout (s)
PLANNER_DELAY_SEC     = 30      # Gazebo stabilisation delay (s)
CLEANUP_WAIT_SEC      = 5
NAV_LOG_INTERVAL      = 5.0     # s between navigation status lines

# FIX-11: per-trial hard ceiling (s); prevents a stuck trial from blocking forever
GLOBAL_TRIAL_TIMEOUT  = 1200    # 20 min absolute max per trial

# FIX-4: early-termination thresholds
EARLY_TERM_STALL_SEC        = 20.0   # abort if no pose update for this long
EARLY_TERM_PROGRESS_SEC     = 30.0   # look-back window for progress check
EARLY_TERM_PROGRESS_THRESH  = 0.02   # min fractional improvement per window
EARLY_TERM_MPC_FAIL_RATE    = 0.7    # abort if MPC success rate < this over last 20 solves

# FIX-7: disk/bag policy
BAG_KEEP_LAST_N_TRIALS       = 5     # keep bags for the N most recent trials (0 = keep all)
BAG_RECORD_ONLY_BEST         = False # if True, only record bags when score improves
BAG_COMPRESS                 = True  # use --compression zstd

# FIX-7: reduced topic set — drop heavy point-cloud from default recording
BAG_TOPICS = [
    "/odom",
    "/go2/pose",
    "/cmd_vel",
    "/a_star/path",
    "/mpc/next_setpoint",
    "/mpc/predicted_path",
    "/mpc/diagnostics",
    "/goal_pose",
    "/tf",
    "/tf_static",
    # "/lidar/points_filtered",  # large — re-enable if needed
]

SCENARIOS = [
    {
        "name": "open_square",
        "world": "default.sdf", "world_pkg": "go2_sim",
        "robot_x": 0.0, "robot_y": 0.0, "robot_heading": 0.0,
        "goals": [[6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
        "obstacles": [
            {"x": 2.5, "y":  0.0},
            {"x": 6.0, "y":  2.5},
            {"x": 3.5, "y":  6.0},
        ],
        "weight": 0.10,
    },
    {
        "name": "open_zigzag",
        "world": "default.sdf", "world_pkg": "go2_sim",
        "robot_x": 0.0, "robot_y": 0.0, "robot_heading": 0.0,
        "goals": [[-4.0, 5.0], [4.0, 5.0], [0.0, 10.0]],
        "obstacles": [
            {"x": -2.0, "y": 2.5},
            {"x":  0.0, "y": 5.0},
            {"x":  2.0, "y": 7.5},
        ],
        "weight": 0.10,
    },
    {
        "name": "warehouse_loop",
        "world": "warehouse.world", "world_pkg": "sim_worlds",
        "robot_x": -10.0, "robot_y": -8.0, "robot_heading": 0.0,
        "goals": [[10.0, -8.0], [10.0, 0.0], [-10.0, 0.0], [-10.0, 8.0]],
        "obstacles": [
            {"x":  0.0, "y": -8.0},
            {"x": 10.0, "y": -4.0},
            {"x":  2.0, "y":  0.0},
            {"x": -6.0, "y":  0.0},
        ],
        "weight": 0.25,
        "timeout": 180,
    },
    {
        "name": "warehouse_cross_aisle",
        "world": "warehouse.world", "world_pkg": "sim_worlds",
        "robot_x": -4.0, "robot_y": 8.0, "robot_heading": -1.5708,
        "goals": [[-4.0, 0.0], [4.0, 0.0], [4.0, -8.0]],
        "obstacles": [
            {"x": -4.0, "y":  4.0},
            {"x":  0.0, "y":  0.5},
            {"x":  4.0, "y": -4.0},
        ],
        "weight": 0.20,
        "timeout": 150,
    },
    {
        "name": "office_traverse",
        "world": "indoor_office.world", "world_pkg": "sim_worlds",
        "robot_x": 2.0, "robot_y": -6.0, "robot_heading": 1.5708,
        "goals": [[4.0, 0.0], [4.0, 3.5], [1.0, 6.0]],
        "obstacles": [
            {"x": 3.0, "y": -3.0},
            {"x": 4.2, "y":  2.0},
            {"x": 2.5, "y":  5.0},
        ],
        "weight": 0.20,
        "timeout": 150,
    },
    {
        "name": "office_corridor",
        "world": "indoor_office.world", "world_pkg": "sim_worlds",
        "robot_x": -4.0, "robot_y": -6.0, "robot_heading": 1.5708,
        "goals": [[-4.0, 0.0], [4.0, 0.0], [4.0, 3.5]],
        "obstacles": [
            {"x": -3.5, "y": -3.0},
            {"x":  0.0, "y": -0.5},
            {"x":  3.8, "y":  2.0},
        ],
        "weight": 0.15,
        "timeout": 150,
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _source_cmd(cmd: str) -> str:
    return f"source {ROS_SETUP} && source {PKG_SETUP} && {cmd}"


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    raise TypeError(f"Not JSON-serialisable: {type(obj)}")


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _ts() -> str:  # FIX-12: type hints throughout
    """ISO timestamp string for log messages."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ─── YAML snapshot builder ────────────────────────────────────────────────────

def build_trial_yaml(base: dict, params: dict, trial_num: int) -> dict:
    trial = copy.deepcopy(base)
    ros_params = trial["/**"]["ros__parameters"]
    for key, val in params.items():
        ros_params[key] = float(val)
    ros_params["_tuning_trial"]     = trial_num
    ros_params["_tuning_timestamp"] = datetime.astimezone(datetime.now()).isoformat() + "Z"
    return trial


# ─── Rosbag recorder (FIX-2, FIX-7) ─────────────────────────────────────────

class RosbagRecorder:
    """Manages a `ros2 bag record` subprocess with clean SIGTERM shutdown."""

    def __init__(self, output_dir: Path, compress: bool = BAG_COMPRESS):
        self.output_dir = output_dir
        self._compress  = compress
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        topics = " ".join(BAG_TOPICS)
        compress_flag = "--compression-mode file --compression-format zstd" if self._compress else ""
        cmd = _source_cmd(
            f"ros2 bag record {topics} {compress_flag} --output {self.output_dir}/bag"
        )
        self._proc = subprocess.Popen(
            ["bash", "-c", cmd],
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.debug("RosbagRecorder started  pid=%d  dir=%s", self._proc.pid, self.output_dir)

    def stop(self) -> None:
        # FIX-2: graceful stop before killing simulation
        if self._proc is None:
            return
        pid = self._proc.pid
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            self._proc.wait(timeout=8)
            log.debug("RosbagRecorder stopped  pid=%d", pid)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            log.warning("RosbagRecorder SIGTERM timed out — sending SIGKILL  pid=%d", pid)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass
        except Exception as exc:
            log.warning("RosbagRecorder stop error  pid=%d  err=%s", pid, exc)
        finally:
            self._proc = None


# ─── Process kill helper (FIX-2) ─────────────────────────────────────────────

def _kill_proc(proc: subprocess.Popen, label: str, sigterm_timeout: float = 8.0) -> None:
    """Send SIGTERM to process group, escalate to SIGKILL on timeout."""
    if proc is None:
        return
    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
        log.debug("Sending SIGTERM to %s  pgid=%d", label, pgid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=sigterm_timeout)
        log.debug("%s exited cleanly", label)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        log.warning("%s SIGTERM timed out — sending SIGKILL  pid=%d", label, pid)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            proc.wait(timeout=3)
        except Exception as exc:
            log.warning("SIGKILL failed for %s  pid=%d  err=%s", label, pid, exc)
    except Exception as exc:
        log.warning("kill_proc error for %s  pid=%d  err=%s", label, pid, exc)


# ─── Simulation manager (FIX-2, FIX-3) ───────────────────────────────────────

class SimulationManager:
    """
    Manages the Gazebo + planner stack.

    FIX-3: One Gazebo instance per trial. Between scenarios we reset robot pose
    and respawn obstacles instead of doing a full restart.
    """

    def __init__(self, gui: bool = False):
        self._proc: Optional[subprocess.Popen] = None
        self._gui  = gui
        self._sim_log_fh = None
        self._current_world: Optional[str] = None

    # ── Launch (once per trial) ───────────────────────────────────────────────

    def launch(self, params_yaml: Path, scenario: dict) -> None:
        """Launch Gazebo + planner stack. Reuse if same world is already running."""
        world_rel = scenario["world"]
        world_pkg = scenario.get("world_pkg", "go2_sim")

        # FIX-3: reuse running simulation when world hasn't changed
        if self._proc is not None and self._current_world == world_rel:
            log.info("  [sim] reusing Gazebo for world=%s  — resetting robot pose", world_rel)
            self._reset_robot(scenario)
            return

        # Need a fresh launch
        if self._proc is not None:
            self.kill()

        pkg_prefix = subprocess.check_output(
            ["bash", "-c", _source_cmd(f"ros2 pkg prefix {world_pkg} 2>/dev/null")],
            text=True,
        ).strip()
        world_path = f"{pkg_prefix}/share/{world_pkg}/worlds/{world_rel}"

        robot_x       = scenario.get("robot_x", 0.0)
        robot_y       = scenario.get("robot_y", 0.0)
        robot_heading = scenario.get("robot_heading", 0.0)
        goals = scenario.get("goals", [[scenario.get("goal_x", 0.0), scenario.get("goal_y", 0.0)]])
        first_gx, first_gy = goals[0]

        cmd = _source_cmd(
            "ros2 launch robot_sim sim_a_star_mpc.launch.py"
            f" gui:={str(self._gui).lower()}"
            f" use_rviz:={str(self._gui).lower()}"
            f" planner_params:={params_yaml}"
            f" wait_for_goal:=false"
            f" goal_x:={first_gx}"
            f" goal_y:={first_gy}"
            f" goal_z:=0.0"
            f" planner_delay_sec:={PLANNER_DELAY_SEC}"
            f" world:={world_path}"
            f" world_init_x:={robot_x}"
            f" world_init_y:={robot_y}"
            f" world_init_heading:={robot_heading}"
        )
        sim_log = RESULTS_DIR / "sim_launch.log"
        sim_log.parent.mkdir(parents=True, exist_ok=True)
        self._sim_log_fh = open(sim_log, "w")
        self._proc = subprocess.Popen(
            ["bash", "-c", cmd],
            preexec_fn=os.setsid,
            stdout=self._sim_log_fh,
            stderr=self._sim_log_fh,
        )
        self._current_world = world_rel

        # FIX-10: condition-based check instead of fixed sleep
        deadline = time.time() + 10
        while time.time() < deadline:
            time.sleep(0.5)
            if self._proc.poll() is not None:
                raise RuntimeError(f"Simulation failed to start — see {sim_log}")
        log.info("  [sim] launched  pid=%d  world=%s", self._proc.pid, world_rel)

    def _reset_robot(self, scenario: dict) -> None:
        """Reset robot to spawn pose via ROS 2 service call."""
        rx = scenario.get("robot_x", 0.0)
        ry = scenario.get("robot_y", 0.0)
        rh = scenario.get("robot_heading", 0.0)
        cmd = _source_cmd(
            f"ros2 service call /reset_robot_pose std_srvs/srv/Empty "
            f"|| ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "
            f"'{{pose: {{pose: {{position: {{x: {rx}, y: {ry}}}, orientation: {{z: {rh}}}}}}}}}'",
        )
        try:
            subprocess.call(
                ["bash", "-c", cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=8,
            )
        except Exception as exc:
            log.warning("  [sim] robot reset failed (non-fatal): %s", exc)

    def spawn_obstacles(self, scenario: dict) -> None:
        """Delete old tuner obstacles, then spawn new ones for this scenario."""
        # Delete previous set (FIX-3: respawn instead of full restart)
        self._delete_obstacles()
        obstacles = scenario.get("obstacles", [])
        for i, obs in enumerate(obstacles):
            model = obs.get("model", "obstacle_cylinder")
            name  = f"tuner_obs_{i}"
            cmd = _source_cmd(
                f"ros2 run sim_scenarios spawn_obstacle"
                f" --name {name} --model {model}"
                f" --x {obs['x']} --y {obs['y']} --z 0.5"
            )
            try:
                subprocess.call(
                    ["bash", "-c", cmd],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=12,
                )
            except subprocess.TimeoutExpired:
                log.warning("  [sim] obstacle spawn timed out  name=%s", name)

    def _delete_obstacles(self) -> None:
        """Delete all tuner_obs_* models from the simulation."""
        for i in range(20):  # generous upper bound
            name = f"tuner_obs_{i}"
            cmd = _source_cmd(
                f"ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "
                f"'{{name: {name}}}' 2>/dev/null"
            )
            try:
                subprocess.call(
                    ["bash", "-c", cmd],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=6,
                )
            except Exception:
                break

    def kill(self) -> None:
        """Terminate simulation completely. Called between trials, not scenarios."""
        # FIX-2: proper SIGTERM→SIGKILL sequence, no pkill -9
        if self._proc is not None:
            _kill_proc(self._proc, "sim_launch")
            self._proc = None
        if self._sim_log_fh:
            try:
                self._sim_log_fh.close()
            except Exception:
                pass
            self._sim_log_fh = None
        self._current_world = None

        # Belt-and-suspenders: targeted SIGTERM for known process names
        _KNOWN_PROCS = [
            "ign gazebo", "gzserver", "gzclient",
            "a_star_node", "mpc_node", "setpoint_to_cmd_vel",
            "odom_to_pose", "cloud_self_filter",
        ]
        for pattern in _KNOWN_PROCS:
            # Send SIGTERM first, not SIGKILL
            subprocess.call(
                ["pkill", "-TERM", "-f", pattern],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        time.sleep(3)
        # Escalate only for survivors
        for pattern in _KNOWN_PROCS:
            subprocess.call(
                ["pkill", "-KILL", "-f", pattern],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        time.sleep(CLEANUP_WAIT_SEC)
        log.debug("  [sim] killed")


# ─── ROS 2 performance monitor (FIX-1, FIX-5) ────────────────────────────────

class PerformanceMonitor(Node):
    """
    Records trajectory, commands, LiDAR closest-obstacle distance, and MPC
    diagnostics during a single scenario.

    FIX-1: Node is created and destroyed per scenario; rclpy itself stays alive.
    FIX-5: PointCloud2 parsed with sensor_msgs_py; falls back to struct on ImportError.
    """

    _DIAG_SUCCESS  = 0
    _DIAG_COST     = 1
    _DIAG_SOLVE_MS = 2
    _DIAG_AVG_MS   = 3
    _DIAG_FAILS    = 4
    _DIAG_SECURITY = 5
    _DIAG_VX_EFF   = 6

    def __init__(self):
        super().__init__("performance_monitor")
        self.trajectory: list     = []
        self.cmd_history: list    = []
        self.mpc_diag: list       = []
        self.predicted_paths: list = []
        self.obs_dist_history: list = []
        self.n_cloud_msgs: int    = 0
        self.min_obs_dist: float  = float("inf")
        self.recording    = False
        self.start_time: Optional[float] = None
        self.goal_pos: Optional[tuple] = None

        _sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(PoseStamped,       "/go2/pose",              self._on_pose,  10)
        self.create_subscription(Twist,             "/cmd_vel",               self._on_cmd,   10)
        self.create_subscription(PointCloud2,       "/lidar/points_filtered", self._on_cloud, _sensor_qos)
        self.create_subscription(Float64MultiArray, "/mpc/diagnostics",       self._on_diag,  10)
        self.create_subscription(NavPath,           "/mpc/predicted_path",    self._on_path,   5)
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

    def publish_goal(self, goal_x: float, goal_y: float) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.pose.position.x = float(goal_x)
        msg.pose.position.y = float(goal_y)
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)
        self.goal_pos = (goal_x, goal_y)

    def start(self, goal_x: float, goal_y: float) -> None:
        self.trajectory.clear()
        self.cmd_history.clear()
        self.mpc_diag.clear()
        self.predicted_paths.clear()
        self.obs_dist_history.clear()
        self.n_cloud_msgs = 0
        self.min_obs_dist = float("inf")
        self.start_time   = time.time()
        self.recording    = True
        self.publish_goal(goal_x, goal_y)

    def stop(self) -> None:
        self.recording = False

    def _elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def _on_pose(self, msg: PoseStamped) -> None:
        if not self.recording:
            return
        q   = msg.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2),
        )
        self.trajectory.append((
            self._elapsed(),
            msg.pose.position.x,
            msg.pose.position.y,
            float(yaw),
        ))

    def _on_cmd(self, msg: Twist) -> None:
        if not self.recording:
            return
        self.cmd_history.append((
            self._elapsed(),
            msg.linear.x, msg.linear.y, msg.angular.z,
        ))

    def _on_diag(self, msg: Float64MultiArray) -> None:
        if not self.recording or len(msg.data) < 7:
            return
        d = msg.data
        self.mpc_diag.append((
            self._elapsed(),
            float(d[self._DIAG_SUCCESS]),
            float(d[self._DIAG_COST]),
            float(d[self._DIAG_SOLVE_MS]),
            float(d[self._DIAG_AVG_MS]),
            float(d[self._DIAG_FAILS]),
            float(d[self._DIAG_SECURITY]),
            float(d[self._DIAG_VX_EFF]),
        ))

    def _on_path(self, msg: NavPath) -> None:
        if not self.recording or not msg.poses:
            return
        last = msg.poses[-1].pose.position
        self.predicted_paths.append((
            self._elapsed(), len(msg.poses), float(last.x), float(last.y),
        ))

    def _on_cloud(self, msg: PointCloud2) -> None:
        """FIX-5: parse PointCloud2 safely via sensor_msgs_py."""
        if not self.recording or msg.width == 0:
            return
        try:
            dists = _parse_pointcloud_dists(msg, max_points=200)
            if dists:
                scan_min = float(min(dists))
                self.min_obs_dist = min(self.min_obs_dist, scan_min)
                self.obs_dist_history.append((self._elapsed(), scan_min))
                self.n_cloud_msgs += 1
        except Exception as exc:
            log.debug("PointCloud2 parse error: %s", exc)


def _parse_pointcloud_dists(msg: PointCloud2, max_points: int = 200) -> list:  # FIX-5
    """
    FIX-5: Extract XY distances from PointCloud2.
    Prefers sensor_msgs_py.point_cloud2.read_points; falls back to struct.unpack.
    """
    try:
        from sensor_msgs_py import point_cloud2 as pc2_utils
        points = list(pc2_utils.read_points(
            msg, field_names=("x", "y"), skip_nans=True,
        ))
        points = points[:max_points]
        return [float(np.hypot(p[0], p[1])) for p in points]
    except ImportError:
        pass
    # Fallback: assume XYZ float layout (original approach)
    import struct
    point_step = msg.point_step
    n = min(msg.width * msg.height, max_points)
    dists = []
    for i in range(n):
        off = i * point_step
        x, y, _ = struct.unpack_from("fff", bytes(msg.data[off:off + 12]))
        dists.append(float(np.hypot(x, y)))
    return dists


# ─── Early termination check (FIX-4) ─────────────────────────────────────────

def _check_early_termination(
    monitor: "PerformanceMonitor",
    current_goal: tuple,
    initial_dist: float,
    now: float,
) -> Optional[str]:
    """
    Returns a reason string if the scenario should be aborted early, else None.
    """
    # 1. No pose updates for too long → navigation stack stalled
    if monitor.trajectory:
        last_pose_t = monitor.trajectory[-1][0]
        if (now - (monitor.start_time or now) - last_pose_t) > EARLY_TERM_STALL_SEC:
            return f"pose stall >{EARLY_TERM_STALL_SEC:.0f}s"
    elif (now - (monitor.start_time or now)) > EARLY_TERM_STALL_SEC:
        return f"no pose data after {EARLY_TERM_STALL_SEC:.0f}s"

    # 2. Progress check over sliding window
    if monitor.trajectory:
        window = [
            p for p in monitor.trajectory
            if (now - (monitor.start_time or now) - p[0]) <= EARLY_TERM_PROGRESS_SEC
        ]
        if len(window) >= 5:
            oldest = window[0]
            newest = window[-1]
            gx, gy = current_goal
            d_old = float(np.hypot(oldest[1] - gx, oldest[2] - gy))
            d_new = float(np.hypot(newest[1] - gx, newest[2] - gy))
            improvement = (d_old - d_new) / max(initial_dist, 0.01)
            if improvement < EARLY_TERM_PROGRESS_THRESH:
                return (
                    f"insufficient progress over {EARLY_TERM_PROGRESS_SEC:.0f}s "
                    f"(Δ={improvement:.3f} < {EARLY_TERM_PROGRESS_THRESH})"
                )

    # 3. MPC failure rate too high in recent window
    if len(monitor.mpc_diag) >= 20:
        recent = monitor.mpc_diag[-20:]
        success_rate = sum(1 for d in recent if d[1] > 0.5) / 20.0
        if success_rate < EARLY_TERM_MPC_FAIL_RATE:
            return f"MPC success rate {success_rate:.0%} < {EARLY_TERM_MPC_FAIL_RATE:.0%}"

    return None


# ─── Score computation (FIX-6) ────────────────────────────────────────────────

def _safe(val: float, default: float = 0.0) -> float:
    """Replace NaN/Inf with a safe default."""
    return default if not np.isfinite(val) else float(val)


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def compute_score(
    monitor: "PerformanceMonitor",
    goal: tuple,
    goals_reached_frac: float = 1.0,
) -> tuple:
    """
    Compute composite score in [0, 1].  FIX-6: NaN guards, clamping, MPC penalty.
    """
    if not monitor.trajectory:
        return 0.0, {"error": "no trajectory"}

    traj     = np.array([(x, y) for _, x, y, _ in monitor.trajectory])
    start    = traj[0]
    final    = traj[-1]
    goal_arr = np.array(goal)

    dist_to_goal  = _safe(float(np.linalg.norm(final - goal_arr)))
    initial_dist  = _safe(float(np.linalg.norm(start - goal_arr)), default=1.0)
    goal_reached  = dist_to_goal < 0.5
    progress_frac = _clamp(_safe(
        max(0.0, (initial_dist - dist_to_goal) / max(initial_dist, 0.01))
    ))

    # Path efficiency
    if len(traj) > 1:
        path_len   = _safe(float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))))
        efficiency = _clamp(_safe(initial_dist / max(path_len, initial_dist)))
    else:
        path_len, efficiency = 0.0, 0.0

    # Control smoothness
    if len(monitor.cmd_history) > 3:
        cmds      = np.array([(vx, vy, wz) for _, vx, vy, wz in monitor.cmd_history])
        mean_jerk = _safe(float(np.mean(np.abs(np.diff(cmds, n=2, axis=0)))), default=999.0)
        smoothness = _clamp(_safe(float(np.exp(-mean_jerk / 2.0))))
    else:
        mean_jerk, smoothness = 0.0, 0.0

    # Obstacle avoidance
    _DANGER_THRESH  = 0.3
    _WARNING_THRESH = 0.6
    obstacle_detected = monitor.n_cloud_msgs >= 5
    if not obstacle_detected:
        obs_avoidance_score = 0.0
        danger_frac = warning_frac = mean_clearance = float("nan")
    else:
        scan_dists     = np.array([d for _, d in monitor.obs_dist_history])
        danger_frac    = _safe(float(np.mean(scan_dists < _DANGER_THRESH)))
        warning_frac   = _safe(float(np.mean(
            (scan_dists >= _DANGER_THRESH) & (scan_dists < _WARNING_THRESH)
        )))
        mean_clearance = _safe(float(np.mean(np.minimum(scan_dists, 2.0))))
        obs_avoidance_score = _clamp(
            (1.0 - danger_frac) * 0.50
            + (1.0 - warning_frac) * 0.30
            + min(mean_clearance / 2.0, 1.0) * 0.20
        )

    # Time efficiency
    if goal_reached and monitor.trajectory:
        elapsed    = monitor.trajectory[-1][0]
        expected   = initial_dist / 0.5
        time_score = _clamp(_safe(expected / max(elapsed, 0.1)))
    else:
        time_score = 0.0

    # FIX-6: explicit MPC success rate penalty
    mpc_success_rate = 1.0
    if monitor.mpc_diag:
        diag = np.array(monitor.mpc_diag)
        mpc_success_rate = _safe(float(diag[:, 1].mean()), default=0.0)
    mpc_penalty = _clamp(mpc_success_rate)  # 0 = all failures, 1 = perfect

    # FIX-6: increase obstacle weight when goal not reached
    if goal_reached:
        score = _clamp(
            0.25 * 1.0
            + 0.15 * _safe(goals_reached_frac)
            + 0.15 * efficiency
            + 0.10 * smoothness
            + 0.20 * obs_avoidance_score
            + 0.10 * time_score
            + 0.05 * mpc_penalty
        )
    else:
        # Higher obstacle weight when goals not reached
        score = _clamp(
            0.18 * _safe(goals_reached_frac)
            + 0.14 * progress_frac
            + 0.07 * efficiency
            + 0.07 * smoothness
            + 0.12 * obs_avoidance_score  # bumped from 0.09
            + 0.07 * mpc_penalty
        )

    metrics: dict[str, Any] = {
        "goal_reached":        goal_reached,
        "goals_reached_frac":  float(_safe(goals_reached_frac)),
        "dist_to_goal":        dist_to_goal,
        "progress_frac":       progress_frac,
        "path_length":         path_len,
        "efficiency":          efficiency,
        "mean_jerk":           mean_jerk,
        "smoothness":          smoothness,
        "min_obs_dist":        float(_safe(monitor.min_obs_dist, default=-1.0)),
        "obstacle_detected":   obstacle_detected,
        "n_cloud_msgs":        monitor.n_cloud_msgs,
        "obs_danger_frac":     danger_frac,
        "obs_warning_frac":    warning_frac,
        "obs_mean_clearance":  mean_clearance,
        "obs_avoidance_score": float(obs_avoidance_score),
        "mpc_success_rate":    mpc_success_rate,
        "mpc_penalty":         mpc_penalty,
        "time_score":          time_score,
        "elapsed_sec":         float(monitor.trajectory[-1][0]) if monitor.trajectory else 0.0,
        "n_traj_points":       len(monitor.trajectory),
        "n_cmd_points":        len(monitor.cmd_history),
        "score":               float(score),
    }

    # MPC diagnostics summary
    if monitor.mpc_diag:
        diag = np.array(monitor.mpc_diag)
        metrics.update({
            "mpc_n_solves":      len(diag),
            "mpc_mean_cost":     _safe(float(diag[:, 2].mean())),
            "mpc_mean_solve_ms": _safe(float(diag[:, 3].mean())),
            "mpc_max_solve_ms":  _safe(float(diag[:, 3].max())),
            "mpc_mean_avg_ms":   _safe(float(diag[:, 4].mean())),
            "mpc_peak_fails":    _safe(float(diag[:, 5].max())),
            "mpc_security_frac": _safe(float(diag[:, 6].mean())),
            "mpc_mean_vx_eff":   _safe(float(diag[:, 7].mean())),
        })
    else:
        metrics["mpc_n_solves"] = 0

    if monitor.predicted_paths:
        metrics["mpc_n_predicted_paths"] = len(monitor.predicted_paths)
        metrics["mpc_mean_horizon_pts"]  = float(
            np.mean([p[1] for p in monitor.predicted_paths])
        )
    else:
        metrics["mpc_n_predicted_paths"] = 0

    return float(score), metrics


# ─── GP surrogate analysis (FIX-8) ───────────────────────────────────────────

# FIX-8: moving average window for parameter importance smoothing
_GP_IMPORTANCE_HISTORY: deque = deque(maxlen=5)

def fit_gp_surrogate(history: list) -> dict:
    """
    FIX-8: Fit ARD Matern-5/2 GP for analysis.
    Requires at least 10 samples (not 3). Catches numerical errors defensively.
    """
    MIN_SAMPLES = 10  # FIX-8: raised from 3
    if len(history) < MIN_SAMPLES:
        return {"skipped": f"need at least {MIN_SAMPLES} observations", "n": len(history)}

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"skipped": "scikit-learn not installed"}

    try:
        X = np.array([[t["params"][p] for p in PARAM_NAMES] for t in history])
        y = np.array([t["score"] for t in history])

        # FIX-8: guard against degenerate y (all same value)
        if np.std(y) < 1e-8:
            return {"skipped": "y variance too low for GP fitting", "n": len(history)}

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        n_dims = X_s.shape[1]

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(
                length_scale=np.ones(n_dims),
                length_scale_bounds=[(1e-2, 1e2)] * n_dims,
                nu=2.5,
            )
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=1e-6,
        )
        gpr.fit(X_s, y)

        fitted        = gpr.kernel_
        matern        = fitted.k1.k2
        constant_val  = float(fitted.k1.k1.constant_value)
        length_scales = matern.length_scale.tolist()
        noise_level   = float(fitted.k2.noise_level)

        inv_ls     = [1.0 / max(ls, 1e-9) for ls in length_scales]
        total_inv  = sum(inv_ls) or 1.0
        raw_sensitivity = {n: float(v / total_inv) for n, v in zip(PARAM_NAMES, inv_ls)}

        # FIX-8: smooth importance with moving average
        _GP_IMPORTANCE_HISTORY.append(raw_sensitivity)
        smoothed_sensitivity = {
            name: float(np.mean([h.get(name, 0.0) for h in _GP_IMPORTANCE_HISTORY]))
            for name in PARAM_NAMES
        }

        best_idx = int(np.argmax(y))
        best_x   = X_s[best_idx:best_idx + 1]
        gp_mean, gp_std = gpr.predict(best_x, return_std=True)

        return {
            "n_observations":    len(history),
            "kernel_theta_log":  fitted.theta.tolist(),
            "constant_value":    constant_val,
            "length_scales":     {n: float(ls) for n, ls in zip(PARAM_NAMES, length_scales)},
            "noise_level":       noise_level,
            "param_sensitivity": smoothed_sensitivity,
            "param_sensitivity_raw": raw_sensitivity,
            "gp_mean_at_best":   float(gp_mean[0]),
            "gp_std_at_best":    float(gp_std[0]),
            "scaler_mean":       scaler.mean_.tolist(),
            "scaler_scale":      scaler.scale_.tolist(),
        }

    except Exception as exc:
        # FIX-8: log warning, don't crash
        log.warning("[GP] fit error (non-fatal): %s", exc)
        return {"error": str(exc), "n": len(history)}


def serialize_tpe_state(trials: Trials, trial_num: int) -> dict:
    losses = [t["result"].get("loss") for t in trials.trials]
    try:
        best_t = trials.best_trial if trials.trials else {}
    except Exception:
        best_t = {}

    return {
        "trial":     trial_num,
        "n_trials":  len(trials.trials),
        "losses":    [float(l) if l is not None else None for l in losses],
        "best_loss": float(best_t.get("result", {}).get("loss", float("inf"))),
        "best_tid":  best_t.get("tid"),
        "Xi": [
            {k: float(v[0]) if v else None for k, v in t["misc"]["vals"].items()}
            for t in trials.trials
            if t["result"].get("status") == STATUS_OK
        ],
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        log.warning("[plot] matplotlib unavailable: %s", exc)
        return None


def _plot_convergence(results: list, out: Path) -> None:
    plt = _get_plt()
    if plt is None:
        return
    scores      = [r["score"] for r in results]
    best_so_far = [max(scores[:i + 1]) for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(1, len(scores) + 1), scores, alpha=0.6, s=30, label="Trial score")
    ax.plot(range(1, len(scores) + 1), best_so_far, "r-", lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("Score"); ax.set_title("MPC Tuning — Convergence")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _plot_param_importance(gp_history: list, out: Path) -> None:
    valid = [s for s in gp_history if "param_sensitivity" in s]
    if not valid:
        return
    plt = _get_plt()
    if plt is None:
        return
    trials = [s["trial"] for s in valid]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in PARAM_NAMES:
        vals = [s["param_sensitivity"][name] for s in valid]
        ax.plot(trials, vals, marker="o", ms=4, label=name)
    ax.set_xlabel("Trial"); ax.set_ylabel("Normalised sensitivity")
    ax.set_title("Parameter Importance from GP Surrogate (smoothed)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_length_scales(gp_history: list, out: Path) -> None:
    valid = [s for s in gp_history if "length_scales" in s]
    if not valid:
        return
    plt = _get_plt()
    if plt is None:
        return
    trials = [s["trial"] for s in valid]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in PARAM_NAMES:
        vals = [s["length_scales"][name] for s in valid]
        ax.semilogy(trials, vals, marker="o", ms=4, label=name)
    ax.set_xlabel("Trial"); ax.set_ylabel("GP length scale (log)")
    ax.set_title("GP Length Scales Evolution")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─── Disk cleanup (FIX-7) ────────────────────────────────────────────────────

def _cleanup_old_bags(results_dir: Path, keep_last_n: int) -> None:
    """Delete rosbag directories from all but the most recent N trials."""
    if keep_last_n <= 0:
        return
    trial_dirs = sorted(results_dir.glob("trial_???"))
    to_prune   = trial_dirs[:-keep_last_n] if len(trial_dirs) > keep_last_n else []
    for td in to_prune:
        for bag_dir in td.rglob("rosbag"):
            if bag_dir.is_dir():
                shutil.rmtree(bag_dir, ignore_errors=True)
                log.debug("[disk] removed bag  %s", bag_dir)


# ─── Main tuner ───────────────────────────────────────────────────────────────

class BayesianMPCTuner:

    def __init__(self, gui: bool = False):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        global log
        log = _setup_logger(RESULTS_DIR)  # FIX-9: structured logging

        self._base_params  = _load_yaml(BASE_PARAMS)
        self._sim          = SimulationManager(gui=gui)
        self._trial_num    = 0
        self._max_evals    = MAX_EVALS
        self._history: list[dict]    = []
        self._gp_history: list[dict] = []
        self._best_score   = -np.inf
        self._best_params: Optional[dict] = None
        self._best_trial   = -1
        self._hp_trials    = Trials()
        self._run_start_t: Optional[float] = None
        self._executor: Optional[SingleThreadedExecutor] = None

    # FIX-1: single rclpy init/shutdown lifecycle
    def _ros_init(self) -> None:
        if not rclpy.ok():
            rclpy.init()
            self._executor = SingleThreadedExecutor()
            log.info("[ROS] rclpy initialised")

    def _ros_shutdown(self) -> None:
        if rclpy.ok():
            rclpy.shutdown()
            log.info("[ROS] rclpy shutdown")

    # ── Per-scenario runner ───────────────────────────────────────────────────

    def _run_scenario(
        self,
        scenario: dict,
        params_yaml: Path,
        trial_dir: Path,
        trial_num: int,
        sc_idx: int,
    ) -> dict:
        sc_name      = scenario["name"]
        scenario_dir = trial_dir / f"scenario_{sc_name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # FIX-9: structured log IDs
        log_prefix = f"[T{trial_num:03d}/SC{sc_idx}/{sc_name}]"
        log.info("%s starting", log_prefix)

        bag = RosbagRecorder(scenario_dir / "rosbag")
        failure_reason: Optional[str] = None

        try:
            # FIX-3: launch reuses Gazebo if world unchanged
            self._sim.launch(params_yaml, scenario)
            bag.start()

            log.info("%s waiting 10s for Gazebo bridge…", log_prefix)
            # FIX-10: interruptible sleep with heartbeat (FIX-9)
            _heartbeat_sleep(10, log_prefix)
            log.info("%s spawning obstacles…", log_prefix)
            self._sim.spawn_obstacles(scenario)

            remaining_delay = max(PLANNER_DELAY_SEC - 10, 5)
            log.info("%s waiting %ds for planner…", log_prefix, remaining_delay)
            _heartbeat_sleep(remaining_delay, log_prefix)

            # FIX-1: create monitor node, add to executor
            monitor = PerformanceMonitor()
            if self._executor:
                self._executor.add_node(monitor)

            goals   = scenario.get("goals", [[scenario.get("goal_x", 0.0), scenario.get("goal_y", 0.0)]])
            timeout = scenario.get("timeout", SCENARIO_TIMEOUT)

            monitor.start(goals[0][0], goals[0][1])
            log.info("%s nav started — %d goal(s), timeout=%ds", log_prefix, len(goals), timeout)

            current_idx    = 0
            goals_reached  = [False] * len(goals)
            end_t          = time.time() + timeout
            # FIX-11: global trial timeout guard
            global_end_t   = time.time() + GLOBAL_TRIAL_TIMEOUT
            last_log_t     = 0.0
            nav_start_t    = time.time()
            initial_dist   = float(np.hypot(
                goals[0][0] - scenario.get("robot_x", 0.0),
                goals[0][1] - scenario.get("robot_y", 0.0),
            ))

            # FIX-10: condition-based loop, no unconditional sleeps
            while time.time() < end_t and time.time() < global_end_t:
                # FIX-1: spin via executor rather than rclpy.spin_once
                if self._executor:
                    self._executor.spin_once(timeout_sec=0.05)
                else:
                    rclpy.spin_once(monitor, timeout_sec=0.05)

                now = time.time()

                # FIX-4: early termination check
                term_reason = _check_early_termination(
                    monitor,
                    tuple(goals[current_idx]),
                    initial_dist,
                    now,
                )
                if term_reason:
                    failure_reason = f"early_term: {term_reason}"
                    log.warning("%s early termination — %s", log_prefix, term_reason)
                    break

                if monitor.trajectory:
                    _, x, y, _ = monitor.trajectory[-1]
                    gx, gy = goals[current_idx]
                    dist = float(np.hypot(x - gx, y - gy))

                    if dist < 0.5:
                        goals_reached[current_idx] = True
                        log.info(
                            "%s goal %d/%d reached  pos=(%.2f,%.2f)  elapsed=%.0fs",
                            log_prefix, current_idx + 1, len(goals), x, y, now - nav_start_t,
                        )
                        current_idx += 1
                        if current_idx >= len(goals):
                            break
                        monitor.publish_goal(goals[current_idx][0], goals[current_idx][1])
                        initial_dist = float(np.hypot(
                            goals[current_idx][0] - x, goals[current_idx][1] - y,
                        ))
                        log.info(
                            "%s → next goal %d/%d: (%.1f,%.1f)",
                            log_prefix, current_idx + 1, len(goals),
                            goals[current_idx][0], goals[current_idx][1],
                        )

                    # FIX-9: periodic heartbeat status
                    if now - last_log_t >= NAV_LOG_INTERVAL:
                        vx = vy = 0.0
                        if monitor.cmd_history:
                            _, vx, vy, _ = monitor.cmd_history[-1]
                        mpc_ok_pct = solve_ms = float("nan")
                        if monitor.mpc_diag:
                            mpc_ok_pct = (
                                sum(1 for d in monitor.mpc_diag[-20:] if d[1] > 0.5)
                                / min(len(monitor.mpc_diag), 20) * 100
                            )
                            solve_ms = monitor.mpc_diag[-1][3]
                        log.info(
                            "%s %.1fs/%.0fs  pos=(%.2f,%.2f)  dist=%.2fm"
                            "  cmd=(%.2f,%.2f)  mpc_ok=%.0f%%  solve=%.1fms",
                            log_prefix, now - nav_start_t, timeout, x, y, dist,
                            vx, vy, mpc_ok_pct, solve_ms,
                        )
                        last_log_t = now

                elif now - last_log_t >= NAV_LOG_INTERVAL:
                    log.info(
                        "%s %.1fs/%.0fs  waiting for pose…",
                        log_prefix, now - nav_start_t, timeout,
                    )
                    last_log_t = now

            if time.time() >= global_end_t:
                failure_reason = failure_reason or "global_trial_timeout"

            goals_reached_frac = sum(goals_reached) / len(goals)
            final_goal = tuple(goals[-1])
            log.info(
                "%s nav done — goals %d/%d (%.0f%%)  elapsed=%.0fs  failure=%s",
                log_prefix, sum(goals_reached), len(goals),
