#!/usr/bin/env bash
# Records essential topics from a running a_star_mpc session.
#
# Usage:
#   ./record_bag.sh [output_dir] [duration_sec]
#
# Example:
#   ./record_bag.sh /tmp/my_run 60
#   ./record_bag.sh                    # defaults below

set -euo pipefail

OUTPUT_DIR="${1:-$(date +%Y%m%d_%H%M%S)_bag}"
DURATION="${2:-}"          # leave empty for unlimited (Ctrl-C to stop)

ROS_SETUP="/opt/ros/jazzy/setup.bash"
if [[ ! -f "$ROS_SETUP" ]]; then
    ROS_SETUP="/opt/ros/humble/setup.bash"
fi
source "$ROS_SETUP"

# Try to source workspace setup if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_SETUP="$SCRIPT_DIR/../install/setup.bash"
if [[ -f "$WORKSPACE_SETUP" ]]; then
    source "$WORKSPACE_SETUP"
fi

TOPICS=(
    /odom
    /go2/pose
    /cmd_vel
    /a_star/path
    /mpc/next_setpoint
    /mpc/predicted_path
    /mpc/diagnostics
    /goal_pose
    /tf
    /tf_static
    /lidar/points_filtered
)

mkdir -p "$OUTPUT_DIR"
echo "Recording to: $OUTPUT_DIR"
echo "Topics: ${TOPICS[*]}"
echo "Press Ctrl-C to stop (or bag will auto-stop after ${DURATION:-unlimited} seconds)"

DURATION_FLAG=""
if [[ -n "$DURATION" ]]; then
    DURATION_FLAG="--duration $DURATION"
fi

ros2 bag record \
    --output "$OUTPUT_DIR/bag" \
    --storage sqlite3 \
    $DURATION_FLAG \
    "${TOPICS[@]}"

echo "Bag saved to: $OUTPUT_DIR/bag"
