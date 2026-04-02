#!/usr/bin/env bash
set -euo pipefail

# Gracefully stop ROS1 nodes first when rosnode is available.
if command -v rosnode >/dev/null 2>&1; then
  rosnode kill -a >/tmp/rosnode_kill_all.log 2>&1 || true
fi

# Collect ROS-related processes and avoid killing this script shell.
self=$$
parent=$PPID
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a pids=()

add_pids() {
  local pid
  while IFS= read -r pid; do
    [ -n "$pid" ] && pids+=("$pid")
  done
}


# Common ROS/Gazebo/CHAMP process names.
for name in \
  roscore rosmaster roslaunch rosout ros2 rviz2 rqt \
  ign ignition gazebo gz gzserver gzclient \
  gzserver-11 gzclient-11 \
  gzserver-fortress gzclient-fortress \
  gazebo gazebo-11 gazebo-fortress \
  parameter_bridge create \
  robot_state_publisher static_transform_publisher \
  pointcloud_to_laserscan_node velocity_limiter \
  quadruped_controller_node state_estimation_node \
  ekf_node spawner \
  champ champ_base champ_msgs champ_gazebo champ_utils champ_localization champ_navigation champ_teleop champ_description; do
  add_pids < <(pgrep -x "$name" || true)
done

# Catch ROS-installed executables that may not use common names.
add_pids < <(pgrep -f '/opt/ros/' || true)
# Catch workspace-built ROS executables.
add_pids < <(pgrep -f "${workspace_root}/install/" || true)
add_pids < <(pgrep -f "${workspace_root}/build/" || true)

# Catch Gazebo Fortress processes by path or args
add_pids < <(pgrep -f 'gazebo.*fortress' || true)
add_pids < <(pgrep -f 'gzserver.*fortress' || true)
add_pids < <(pgrep -f 'gzclient.*fortress' || true)
add_pids < <(pgrep -f 'ign gazebo' || true)
add_pids < <(pgrep -f 'ignition-gazebo' || true)
add_pids < <(pgrep -f 'gz sim' || true)

# Catch ROS 2 launch + bridge processes for the new Go2 sim stack.
add_pids < <(pgrep -f 'sim_champ.launch.py' || true)
add_pids < <(pgrep -f 'ros_ign_gazebo' || true)
add_pids < <(pgrep -f 'ros_ign_bridge' || true)
add_pids < <(pgrep -f 'controller_manager/spawner' || true)
add_pids < <(pgrep -f 'pointcloud_to_laserscan' || true)
add_pids < <(pgrep -f 'velocity_limiter' || true)

# Catch CHAMP python nodes/scripts
add_pids < <(pgrep -f 'champ' || true)

# Unique + filter out this script shell and parent.
pids=$(
  printf '%s\n' "${pids[@]:-}" \
  | grep -Ev '^\s*$' \
  | sort -u \
  | grep -Ev "^(${self}|${parent})$" \
  || true
)

if [ -z "${pids:-}" ]; then
  echo "No ROS processes found."
  exit 0
fi

echo "Stopping ROS processes:"
printf '%s\n' "$pids"

# Try TERM first.
while IFS= read -r pid; do
  kill "$pid" 2>/dev/null || true
done <<< "$pids"

sleep 1

# Force kill any remaining.
while IFS= read -r pid; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
done <<< "$pids"

echo "Done."
