#pragma once

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include <graphnav_msgs/msg/key_value.hpp>
#include <graphnav_msgs/msg/navigation_graph.hpp>
#include <graphnav_msgs/msg/node.hpp>
#include <graphnav_msgs/msg/uuid.hpp>

#include <iomanip>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wildos_graphnav_planner
{

inline std::string uuid_to_string(const graphnav_msgs::msg::UUID& uuid)
{
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) oss << '-';
    oss << std::setw(2) << static_cast<int>(uuid.id[i]);
  }
  return oss.str();
}

class UnexploredSpaceMap
{
public:
  UnexploredSpaceMap(double min_x, double max_x, double min_y, double max_y, double margin, double resolution);
  void mark_explored(double x, double y, double radius);
  void compute_distance_from(double x, double y);
  double query_distance_to(double x, double y, int radius = 1);

private:
  bool in_bounds(int x, int y) const;

  size_t size_x_;
  size_t size_y_;
  double origin_x_;
  double origin_y_;
  double resolution_;
  Eigen::MatrixXi map_;
  Eigen::MatrixXf dist_map_;
};

class Planner
{
public:
  Planner(rclcpp::Logger logger);
  void set_trav_class(const std::string& trav_class);
  void update_graph(graphnav_msgs::msg::NavigationGraph::ConstSharedPtr graph);
  std::vector<Eigen::Vector3d> plan_to_goal(Eigen::Vector3d& goal, double goal_radius, rclcpp::Time current_time);

  double frontier_dist_cost_factor_ = 2.0;
  double goal_dist_cost_factor_ = 1.0;
  double frontier_score_factor_ = 10.0;
  double min_local_frontier_score_ = 0.4;
  double local_frontier_radius_ = 7.0;
  double path_smoothness_period_ = 10.0;

private:
  using Adjacency = std::vector<std::vector<std::pair<size_t, double>>>;

  std::optional<UnexploredSpaceMap> compute_unexplored_space_map();
  std::vector<size_t> dijkstra_shortest_path(size_t start_idx, size_t goal_idx, const Adjacency& adj) const;
  static bool has_key(const std::vector<graphnav_msgs::msg::KeyValue>& props, const std::string& key);
  static std::vector<float> get_values(const std::vector<graphnav_msgs::msg::KeyValue>& props, const std::string& key);

  rclcpp::Logger logger_;
  std::string trav_class_;

  std::vector<graphnav_msgs::msg::Node> nodes_;
  Adjacency adjacency_;
  size_t current_node_idx_ = 0;
  size_t trav_class_idx_ = 0;
  std::optional<UnexploredSpaceMap> unexplored_space_map_;
  std::optional<Eigen::Vector3d> latest_frontier_;
  std::optional<rclcpp::Time> latest_frontier_time_;

  std::unordered_map<size_t, std::pair<graphnav_msgs::msg::Node, std::pair<double, double>>> frontier_scores_;
};

}  // namespace wildos_graphnav_planner
