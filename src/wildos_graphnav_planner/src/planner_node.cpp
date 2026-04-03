#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <optional>

#include "wildos_graphnav_planner/planner.hpp"

namespace wildos_graphnav_planner
{

class PlannerNode : public rclcpp::Node
{
public:
  explicit PlannerNode(const rclcpp::NodeOptions& options)
  : Node("planner_node", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_, this), planner_(this->get_logger())
  {
    declare_parameter("frontier_dist_cost_factor", 2.0);
    declare_parameter("goal_dist_cost_factor", 1.0);
    declare_parameter("frontier_score_factor", 10.0);
    declare_parameter("min_local_frontier_score", 0.4);
    declare_parameter("local_frontier_radius", 7.0);
    declare_parameter("path_smoothness_period", 10.0);
    declare_parameter("trav_class", "default");
    declare_parameter("goal_radius", 3.0);

    planner_.frontier_dist_cost_factor_ = get_parameter("frontier_dist_cost_factor").as_double();
    planner_.goal_dist_cost_factor_ = get_parameter("goal_dist_cost_factor").as_double();
    planner_.frontier_score_factor_ = get_parameter("frontier_score_factor").as_double();
    planner_.min_local_frontier_score_ = get_parameter("min_local_frontier_score").as_double();
    planner_.local_frontier_radius_ = get_parameter("local_frontier_radius").as_double();
    planner_.path_smoothness_period_ = get_parameter("path_smoothness_period").as_double();
    planner_.set_trav_class(get_parameter("trav_class").as_string());
    goal_radius_ = get_parameter("goal_radius").as_double();

    graph_sub_ = create_subscription<graphnav_msgs::msg::NavigationGraph>(
      "~/nav_graph", 10,
      [this](const graphnav_msgs::msg::NavigationGraph::ConstSharedPtr msg) {
        planner_.update_graph(msg);
        latest_graph_header_ = msg->header;
        plan_to_goal();
      });

    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/goal_pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
        goal_pose_ = msg;
        plan_to_goal();
      });

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "~/odom", 10, [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { odom_ = msg; });

    path_pub_ = create_publisher<nav_msgs::msg::Path>("~/path", 10);
  }

private:
  void plan_to_goal()
  {
    if (!goal_pose_ || !latest_graph_header_) {
      return;
    }

    geometry_msgs::msg::PoseStamped goal = *goal_pose_;
    goal.header.stamp = latest_graph_header_->stamp;

    geometry_msgs::msg::PoseStamped goal_in_graph_frame;
    try {
      goal_in_graph_frame = tf_buffer_.transform(goal, latest_graph_header_->frame_id, tf2::durationFromSec(0.1));
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(get_logger(), "Could not transform goal pose to graph frame: %s", ex.what());
      return;
    }

    Eigen::Vector3d goal_vec(goal_in_graph_frame.pose.position.x, goal_in_graph_frame.pose.position.y, goal_in_graph_frame.pose.position.z);
    auto path = planner_.plan_to_goal(goal_vec, goal_radius_, get_clock()->now());

    nav_msgs::msg::Path path_msg;
    path_msg.header = *latest_graph_header_;
    path_msg.poses.resize(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
      path_msg.poses[i].header = path_msg.header;
      path_msg.poses[i].pose.position.x = path[i].x();
      path_msg.poses[i].pose.position.y = path[i].y();
      path_msg.poses[i].pose.position.z = path[i].z();
      if (i < path.size() - 1) {
        Eigen::Vector3d d = (path[i + 1] - path[i]).normalized();
        Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
        m.col(1) = Eigen::Vector3d::UnitZ().cross(d).normalized();
        m.col(0) = m.col(1).cross(m.col(2)).normalized();
        Eigen::Quaterniond q(m);
        path_msg.poses[i].pose.orientation.x = q.x();
        path_msg.poses[i].pose.orientation.y = q.y();
        path_msg.poses[i].pose.orientation.z = q.z();
        path_msg.poses[i].pose.orientation.w = q.w();
      } else {
        path_msg.poses[i].pose.orientation = goal_in_graph_frame.pose.orientation;
      }
    }
    path_pub_->publish(path_msg);

    if (odom_) {
      try {
        geometry_msgs::msg::PoseStamped goal_in_odom_frame = tf_buffer_.transform(goal, odom_->header.frame_id, tf2::durationFromSec(0.1));
        Eigen::Vector3d goal_odom(goal_in_odom_frame.pose.position.x, goal_in_odom_frame.pose.position.y, goal_in_odom_frame.pose.position.z);
        Eigen::Vector3d odom_vec(odom_->pose.pose.position.x, odom_->pose.pose.position.y, odom_->pose.pose.position.z);
        if ((goal_odom - odom_vec).norm() < goal_radius_) {
          goal_pose_.reset();
        }
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(get_logger(), "Could not transform goal pose to odom frame: %s", ex.what());
      }
    }
  }

  rclcpp::Subscription<graphnav_msgs::msg::NavigationGraph>::SharedPtr graph_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  geometry_msgs::msg::PoseStamped::ConstSharedPtr goal_pose_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  std::optional<std_msgs::msg::Header> latest_graph_header_;
  double goal_radius_;
  Planner planner_;
};

}  // namespace wildos_graphnav_planner

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(wildos_graphnav_planner::PlannerNode)
