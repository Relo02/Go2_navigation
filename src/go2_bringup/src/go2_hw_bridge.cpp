/**
 * go2_hw_bridge.cpp
 *
 * Central hardware bridge between the Go2 robot and the Nav2 navigation stack.
 * This is the only node in the stack that speaks the Unitree Go2 protocol.
 *
 * ── Subscriptions ────────────────────────────────────────────────────────────
 *  /sportmodestate  [unitree_go/SportModeState]
 *    Published by the Go2 internal ROS2 bridge at ~50 Hz.
 *    Contains proprioceptive state: position, velocity, IMU, battery.
 *
 *  /cmd_vel  [geometry_msgs/Twist]
 *    Velocity commands from Nav2's controller_server.
 *
 * ── Publications ─────────────────────────────────────────────────────────────
 *  /odom              [nav_msgs/Odometry]       — re-published for Nav2
 *  /robot_status      [robot_common_interfaces/RobotStatus] — health at 1 Hz
 *  /api/sport/request [unitree_api/Request]     — motion commands to Go2
 *  TF: odom → base_footprint  (2D pose from SportModeState)
 *
 * ── TF Tree ──────────────────────────────────────────────────────────────────
 *  map  ──(slam_toolbox)──►  odom  ──(this node)──►  base_footprint
 *                                                           └─(URDF)─►  base_link
 *                                                                            └─►  lidar_link
 *
 * ── Design Notes ─────────────────────────────────────────────────────────────
 *  - The Go2 SportModeState.position is in the Go2's body-initialised internal
 *    frame.  We treat this frame as our 'odom' frame (short-term accurate,
 *    accumulates drift over time — slam_toolbox corrects this via map→odom).
 *  - Z is zeroed in the odom→base_footprint transform (2D navigation convention).
 *  - A cmd_vel watchdog publishes StopMove if no command arrives within
 *    cmd_timeout seconds.  This prevents the robot from running away if
 *    Nav2 crashes.
 *  - The node clamps velocities to safe limits before forwarding to the robot.
 *
 * TODO(agibot): For AgiBot D1 Ultra, create agibot_d1_bringup/src/d1_hw_bridge.cpp
 *   following the same pattern:  subscribe to D1 state, publish odom + TF,
 *   subscribe to cmd_vel, forward via D1's API.
 *   Zero changes required in robot_nav.
 */

#include <algorithm>
#include <chrono>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/transform_broadcaster.h"

#include "unitree_api/msg/request.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "robot_common_interfaces/msg/robot_status.hpp"
#include "go2_bringup/sport_client.hpp"

namespace go2_bringup
{

class Go2HwBridge : public rclcpp::Node
{
public:
  Go2HwBridge()
  : Node("go2_hw_bridge")
  {
    // ── Parameters ────────────────────────────────────────────────────────
    declare_parameter<std::string>("state_topic",      "/sportmodestate");
    declare_parameter<std::string>("lowstate_topic",   "/lowstate");
    declare_parameter<std::string>("cmd_vel_topic",    "/cmd_vel");
    declare_parameter<std::string>("sport_req_topic",  "/api/sport/request");
    declare_parameter<std::string>("odom_topic",       "odom");
    declare_parameter<std::string>("odom_frame",       "odom");
    declare_parameter<std::string>("base_frame",       "base_footprint");
    declare_parameter<bool>("publish_tf",              true);
    declare_parameter<double>("cmd_timeout",           0.5);   // [s] watchdog
    declare_parameter<double>("max_linear_vel",        0.8);   // [m/s]
    declare_parameter<double>("max_lateral_vel",       0.4);   // [m/s]
    declare_parameter<double>("max_angular_vel",       1.5);   // [rad/s]

    state_topic_    = get_parameter("state_topic").as_string();
    lowstate_topic_ = get_parameter("lowstate_topic").as_string();
    cmd_vel_topic_  = get_parameter("cmd_vel_topic").as_string();
    sport_topic_    = get_parameter("sport_req_topic").as_string();
    odom_topic_     = get_parameter("odom_topic").as_string();
    odom_frame_     = get_parameter("odom_frame").as_string();
    base_frame_     = get_parameter("base_frame").as_string();
    publish_tf_     = get_parameter("publish_tf").as_bool();
    cmd_timeout_    = get_parameter("cmd_timeout").as_double();
    max_lin_        = get_parameter("max_linear_vel").as_double();
    max_lat_        = get_parameter("max_lateral_vel").as_double();
    max_ang_        = get_parameter("max_angular_vel").as_double();

    // ── Publishers ────────────────────────────────────────────────────────
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    odom_pub_   = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, rclcpp::QoS(10));
    status_pub_ = create_publisher<robot_common_interfaces::msg::RobotStatus>(
      "/robot_status", rclcpp::QoS(10).transient_local());
    sport_pub_  = create_publisher<unitree_api::msg::Request>(
      sport_topic_, rclcpp::QoS(10));

    // ── Subscriptions ─────────────────────────────────────────────────────
    state_sub_ = create_subscription<unitree_go::msg::SportModeState>(
      state_topic_, rclcpp::QoS(10),
      std::bind(&Go2HwBridge::on_sport_state, this, std::placeholders::_1));

    lowstate_sub_ = create_subscription<unitree_go::msg::LowState>(
      lowstate_topic_, rclcpp::QoS(10),
      std::bind(&Go2HwBridge::on_low_state, this, std::placeholders::_1));

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::QoS(10),
      std::bind(&Go2HwBridge::on_cmd_vel, this, std::placeholders::_1));

    // ── Watchdog timer ────────────────────────────────────────────────────
    // Fires at 10 Hz.  If cmd_vel goes stale, sends StopMove to the robot.
    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&Go2HwBridge::on_watchdog, this));

    // ── Status timer ─────────────────────────────────────────────────────
    status_timer_ = create_wall_timer(
      std::chrono::milliseconds(1000),
      std::bind(&Go2HwBridge::publish_status, this));

    RCLCPP_INFO(
      get_logger(),
      "Go2HwBridge ready.  state=%s  lowstate=%s  cmd=%s  odom=%s  sport=%s",
      state_topic_.c_str(), lowstate_topic_.c_str(),
      cmd_vel_topic_.c_str(), odom_topic_.c_str(), sport_topic_.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  limits: lin=%.2f lat=%.2f ang=%.2f  watchdog=%.1fs  publish_tf=%s",
      max_lin_, max_lat_, max_ang_, cmd_timeout_, publish_tf_ ? "true" : "false");
  }

private:
  // ── Sport state callback ─────────────────────────────────────────────────

  void on_sport_state(const unitree_go::msg::SportModeState::SharedPtr msg)
  {
    // Validate the Go2 hardware clock against a minimum wall-clock epoch.
    // The Go2 clock is a wall clock when NTP-synced, but during early boot it
    // may report device uptime (e.g. 300 s since boot → Jan 1970 + 5 min),
    // which would corrupt TF/SLAM timestamps.
    // kMinEpochSec = 2020-01-01T00:00:00Z.  Uptime values are always smaller.
    static constexpr int64_t kMinEpochSec = 1577836800LL;
    rclcpp::Time stamp;
    if (static_cast<int64_t>(msg->stamp.sec) >= kMinEpochSec) {
      stamp = rclcpp::Time(
        static_cast<int32_t>(msg->stamp.sec),
        msg->stamp.nanosec,
        RCL_ROS_TIME);
    } else {
      stamp = now();
    }

    // ── Odometry message ──────────────────────────────────────────────────
    // The Go2's SportModeState.position is in the robot's internal odometry frame.
    // We treat this as our 'odom' frame.
    // Z = 0: 2D navigation convention (slam_toolbox operates in the XY plane).
    nav_msgs::msg::Odometry odom;
    odom.header.stamp    = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id  = base_frame_;

    odom.pose.pose.position.x = static_cast<double>(msg->position[0]);
    odom.pose.pose.position.y = static_cast<double>(msg->position[1]);
    odom.pose.pose.position.z = 0.0;  // 2D: always zero

    // IMU quaternion from Go2 is [w, x, y, z]
    odom.pose.pose.orientation.w = static_cast<double>(msg->imu_state.quaternion[0]);
    odom.pose.pose.orientation.x = static_cast<double>(msg->imu_state.quaternion[1]);
    odom.pose.pose.orientation.y = static_cast<double>(msg->imu_state.quaternion[2]);
    odom.pose.pose.orientation.z = static_cast<double>(msg->imu_state.quaternion[3]);

    // Body-frame velocities from Go2 state
    odom.twist.twist.linear.x  = static_cast<double>(msg->velocity[0]);
    odom.twist.twist.linear.y  = static_cast<double>(msg->velocity[1]);
    odom.twist.twist.linear.z  = 0.0;
    odom.twist.twist.angular.z = static_cast<double>(msg->yaw_speed);

    odom_pub_->publish(odom);

    if (publish_tf_) {
      // ── TF: odom → base_footprint ──────────────────────────────────────
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp    = stamp;
      tf.header.frame_id = odom_frame_;
      tf.child_frame_id  = base_frame_;

      tf.transform.translation.x = odom.pose.pose.position.x;
      tf.transform.translation.y = odom.pose.pose.position.y;
      tf.transform.translation.z = 0.0;
      tf.transform.rotation      = odom.pose.pose.orientation;

      tf_broadcaster_->sendTransform(tf);
    }

    // Cache state for status publisher
    last_state_ = *msg;
    has_state_  = true;
  }

  // ── LowState callback (battery data) ────────────────────────────────────

  void on_low_state(const unitree_go::msg::LowState::SharedPtr msg)
  {
    last_low_state_ = *msg;
    has_low_state_  = true;
  }

  // ── cmd_vel callback ─────────────────────────────────────────────────────

  void on_cmd_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    last_cmd_time_ = now();

    // Clamp to safe limits
    const float vx   = clamp(static_cast<float>(msg->linear.x),  -max_lin_, max_lin_);
    const float vy   = clamp(static_cast<float>(msg->linear.y),  -max_lat_, max_lat_);
    const float vyaw = clamp(static_cast<float>(msg->angular.z), -max_ang_, max_ang_);

    unitree_api::msg::Request req;
    sport_.Move(req, vx, vy, vyaw);
    sport_pub_->publish(req);
  }

  // ── Watchdog ──────────────────────────────────────────────────────────────

  void on_watchdog()
  {
    if (!last_cmd_time_.has_value()) {
      return;  // No command ever received; don't send spurious stops
    }

    const double age = (now() - *last_cmd_time_).seconds();
    if (age > cmd_timeout_) {
      if (!stopped_) {
        unitree_api::msg::Request req;
        sport_.StopMove(req);
        sport_pub_->publish(req);
        stopped_ = true;
        RCLCPP_WARN(get_logger(),
          "cmd_vel stale (%.2fs > %.2fs) — sent StopMove", age, cmd_timeout_);
      }
    } else {
      stopped_ = false;
    }
  }

  // ── Status publisher ──────────────────────────────────────────────────────

  void publish_status()
  {
    robot_common_interfaces::msg::RobotStatus status;
    status.header.stamp = now();

    if (has_state_) {
      status.hw_ok       = true;
      status.motion_mode = std::to_string(last_state_.mode);
    } else {
      status.hw_ok = false;
    }

    // Battery state comes from /lowstate (BmsState), not from SportModeState.
    // Publish 0 / warn once until the first LowState message arrives.
    if (has_low_state_) {
      // BmsState.soc is integer [0..100 %] → normalise to [0.0..1.0]
      status.battery_soc     = static_cast<float>(last_low_state_.bms_state.soc) / 100.0f;
      status.battery_voltage = last_low_state_.power_v;
    } else {
      status.battery_soc     = 0.0f;
      status.battery_voltage = 0.0f;
      if (has_state_) {
        // Warn once: LowState not yet received (topic may not be published).
        RCLCPP_WARN_ONCE(get_logger(),
          "No /lowstate received yet — battery_soc and battery_voltage are 0 "
          "(check that the Go2 bridge publishes %s)", lowstate_topic_.c_str());
      }
    }
    status.estop_active = stopped_;

    status_pub_->publish(status);
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  static float clamp(float v, double lo, double hi)
  {
    return static_cast<float>(std::max(lo, std::min(hi, static_cast<double>(v))));
  }

  // ── Members ───────────────────────────────────────────────────────────────

  std::string state_topic_;
  std::string lowstate_topic_;
  std::string cmd_vel_topic_;
  std::string sport_topic_;
  std::string odom_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  bool publish_tf_;
  double cmd_timeout_;
  double max_lin_, max_lat_, max_ang_;

  SportClient sport_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<robot_common_interfaces::msg::RobotStatus>::SharedPtr status_pub_;
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr sport_pub_;

  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr state_sub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  std::optional<rclcpp::Time> last_cmd_time_;
  bool stopped_       = false;
  bool has_state_     = false;
  bool has_low_state_ = false;
  unitree_go::msg::SportModeState last_state_;
  unitree_go::msg::LowState       last_low_state_;
};

}  // namespace go2_bringup

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<go2_bringup::Go2HwBridge>());
  rclcpp::shutdown();
  return 0;
}
