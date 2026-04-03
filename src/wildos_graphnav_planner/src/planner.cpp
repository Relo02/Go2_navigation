#include "wildos_graphnav_planner/planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace wildos_graphnav_planner
{

UnexploredSpaceMap::UnexploredSpaceMap(double min_x, double max_x, double min_y, double max_y, double margin, double resolution)
: resolution_(resolution)
{
  origin_x_ = min_x - margin;
  origin_y_ = min_y - margin;
  size_x_ = static_cast<size_t>(std::ceil((max_x - min_x + 2 * margin) / resolution_));
  size_y_ = static_cast<size_t>(std::ceil((max_y - min_y + 2 * margin) / resolution_));
  map_ = Eigen::MatrixXi::Ones(size_x_, size_y_);
}

void UnexploredSpaceMap::mark_explored(double x, double y, double radius)
{
  int ix = static_cast<int>((x - origin_x_) / resolution_);
  int iy = static_cast<int>((y - origin_y_) / resolution_);
  int ir = static_cast<int>(std::ceil(radius / resolution_));
  for (int dx = -ir; dx <= ir; ++dx) {
    for (int dy = -ir; dy <= ir; ++dy) {
      if (std::hypot(dx * resolution_, dy * resolution_) <= radius) {
        int nx = ix + dx;
        int ny = iy + dy;
        if (in_bounds(nx, ny)) map_(nx, ny) = 0;
      }
    }
  }
}

void UnexploredSpaceMap::compute_distance_from(double x, double y)
{
  int ix = static_cast<int>((x - origin_x_) / resolution_);
  int iy = static_cast<int>((y - origin_y_) / resolution_);
  Eigen::Vector2i goal(ix, iy);
  dist_map_ = Eigen::MatrixXf::Constant(size_x_, size_y_, std::numeric_limits<float>::infinity());

  using Cell = std::pair<int, int>;
  std::priority_queue<std::pair<float, Cell>, std::vector<std::pair<float, Cell>>, std::greater<>> pq;

  if (in_bounds(ix, iy)) {
    dist_map_(ix, iy) = 0.0f;
    pq.emplace(0.0f, Cell(ix, iy));
  } else {
    for (int x_idx = 0; x_idx < static_cast<int>(size_x_); ++x_idx) {
      Eigen::Vector2i p(x_idx, (goal.y() < 0) ? 0 : static_cast<int>(size_y_) - 1);
      dist_map_(p.x(), p.y()) = (p - goal).norm() * resolution_;
      pq.emplace(dist_map_(p.x(), p.y()), Cell(p.x(), p.y()));
    }
    for (int y_idx = 0; y_idx < static_cast<int>(size_y_); ++y_idx) {
      Eigen::Vector2i p((goal.x() < 0) ? 0 : static_cast<int>(size_x_) - 1, y_idx);
      dist_map_(p.x(), p.y()) = (p - goal).norm() * resolution_;
      pq.emplace(dist_map_(p.x(), p.y()), Cell(p.x(), p.y()));
    }
  }

  const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  const float cost[8] = {1, std::sqrt(2), 1, std::sqrt(2), 1, std::sqrt(2), 1, std::sqrt(2)};

  while (!pq.empty()) {
    auto [cur_dist, cell] = pq.top();
    pq.pop();
    int cx = cell.first;
    int cy = cell.second;
    if (cur_dist > dist_map_(cx, cy)) continue;
    for (int dir = 0; dir < 8; ++dir) {
      int nx = cx + dx[dir];
      int ny = cy + dy[dir];
      if (!in_bounds(nx, ny) || map_(nx, ny) == 0) continue;
      float new_dist = dist_map_(cx, cy) + cost[dir] * resolution_;
      if (new_dist < dist_map_(nx, ny)) {
        dist_map_(nx, ny) = new_dist;
        pq.emplace(new_dist, Cell(nx, ny));
      }
    }
  }
}

double UnexploredSpaceMap::query_distance_to(double x, double y, int radius)
{
  int ix = static_cast<int>((x - origin_x_) / resolution_);
  int iy = static_cast<int>((y - origin_y_) / resolution_);
  float min_dist = std::numeric_limits<float>::infinity();
  for (int dx = -radius; dx <= radius; ++dx) {
    for (int dy = -radius; dy <= radius; ++dy) {
      int nx = ix + dx;
      int ny = iy + dy;
      if (in_bounds(nx, ny)) min_dist = std::min(min_dist, dist_map_(nx, ny));
    }
  }
  return min_dist;
}

bool UnexploredSpaceMap::in_bounds(int x, int y) const
{
  return x >= 0 && y >= 0 && x < static_cast<int>(size_x_) && y < static_cast<int>(size_y_);
}

Planner::Planner(rclcpp::Logger logger) : logger_(logger) {}

void Planner::set_trav_class(const std::string& trav_class)
{
  trav_class_ = trav_class;
}

bool Planner::has_key(const std::vector<graphnav_msgs::msg::KeyValue>& props, const std::string& key)
{
  return std::any_of(props.begin(), props.end(), [&](const auto& kv) { return kv.key == key; });
}

std::vector<float> Planner::get_values(const std::vector<graphnav_msgs::msg::KeyValue>& props, const std::string& key)
{
  for (const auto& kv : props) {
    if (kv.key == key) return kv.value;
  }
  return {};
}

void Planner::update_graph(graphnav_msgs::msg::NavigationGraph::ConstSharedPtr graph)
{
  nodes_ = graph->nodes;
  adjacency_.assign(nodes_.size(), {});

  const auto trav_class_it = std::find(graph->trav_classes.begin(), graph->trav_classes.end(), trav_class_);
  if (trav_class_it == graph->trav_classes.end()) {
    RCLCPP_WARN(logger_, "Traversability class %s not found in graph", trav_class_.c_str());
    trav_class_idx_ = 0;
  } else {
    trav_class_idx_ = static_cast<size_t>(std::distance(graph->trav_classes.begin(), trav_class_it));
  }

  for (const auto& edge : graph->edges) {
    if (trav_class_idx_ < edge.traversability.size()) {
      const double weight = edge.traversability[trav_class_idx_].traversability_cost;
      if (edge.from_idx < adjacency_.size() && edge.to_idx < adjacency_.size()) {
        adjacency_[edge.from_idx].push_back({edge.to_idx, weight});
        adjacency_[edge.to_idx].push_back({edge.from_idx, weight});
      }
    }
  }

  current_node_idx_ = graph->current_node_idx;
  unexplored_space_map_ = compute_unexplored_space_map();
}

std::optional<UnexploredSpaceMap> Planner::compute_unexplored_space_map()
{
  if (nodes_.empty()) return std::nullopt;

  double min_x = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();

  for (const auto& node : nodes_) {
    const auto& pos = node.pose.position;
    min_x = std::min(min_x, pos.x);
    max_x = std::max(max_x, pos.x);
    min_y = std::min(min_y, pos.y);
    max_y = std::max(max_y, pos.y);
  }

  constexpr double resolution = 1.0;
  constexpr double margin = 10.0;
  UnexploredSpaceMap map(min_x, max_x, min_y, max_y, margin, resolution);
  for (const auto& node : nodes_) {
    if (trav_class_idx_ < node.trav_properties.size()) {
      const double explored_radius = node.trav_properties[trav_class_idx_].explored_radius;
      if (explored_radius > 0) {
        const auto& pos = node.pose.position;
        map.mark_explored(pos.x, pos.y, explored_radius);
      }
    }
  }
  return map;
}

std::vector<size_t> Planner::dijkstra_shortest_path(size_t start_idx, size_t goal_idx, const Adjacency& adj) const
{
  struct QNode {
    double d;
    size_t idx;
    bool operator>(const QNode& other) const { return d > other.d; }
  };

  std::vector<double> dist(adj.size(), std::numeric_limits<double>::infinity());
  std::vector<size_t> prev(adj.size(), static_cast<size_t>(-1));
  std::priority_queue<QNode, std::vector<QNode>, std::greater<>> pq;

  dist[start_idx] = 0.0;
  pq.push({0.0, start_idx});

  while (!pq.empty()) {
    const auto [cur_d, u] = pq.top();
    pq.pop();
    if (cur_d > dist[u]) continue;
    if (u == goal_idx) break;
    for (const auto& [v, w] : adj[u]) {
      const double nd = cur_d + w;
      if (nd < dist[v]) {
        dist[v] = nd;
        prev[v] = u;
        pq.push({nd, v});
      }
    }
  }

  if (!std::isfinite(dist[goal_idx])) return {};

  std::vector<size_t> path;
  for (size_t cur = goal_idx; cur != static_cast<size_t>(-1); cur = prev[cur]) {
    path.push_back(cur);
    if (cur == start_idx) break;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<Eigen::Vector3d> Planner::plan_to_goal(Eigen::Vector3d& goal, double goal_radius, rclcpp::Time current_time)
{
  if (!unexplored_space_map_) return {};

  unexplored_space_map_->compute_distance_from(goal.x(), goal.y());

  graphnav_msgs::msg::Node virtual_goal_node;
  virtual_goal_node.pose.position.x = goal.x();
  virtual_goal_node.pose.position.y = goal.y();
  virtual_goal_node.pose.position.z = goal.z();

  Adjacency graph = adjacency_;
  const size_t virtual_goal = graph.size();
  graph.emplace_back();

  frontier_scores_.clear();
  std::vector<size_t> local_scored_frontiers;
  bool is_scored_graph = true;

  for (size_t id = 0; id < nodes_.size(); ++id) {
    const auto& node = nodes_[id];
    Eigen::Vector3d node_pos(node.pose.position.x, node.pose.position.y, node.pose.position.z);

    if (trav_class_idx_ < node.trav_properties.size() && node.trav_properties[trav_class_idx_].is_frontier) {
      double frontier_path_distance = std::numeric_limits<double>::max();
      double frontier_score = -1.0;
      double frontier_cost = std::numeric_limits<double>::max();

      for (const auto& frontier_pt : node.trav_properties[trav_class_idx_].frontier_points) {
        frontier_path_distance = std::min(frontier_path_distance, unexplored_space_map_->query_distance_to(frontier_pt.x, frontier_pt.y));
      }

      bool has_frontier_scores = false;
      double cur_frontier_dist_cost_factor = std::numeric_limits<double>::max();
      if (has_key(node.properties, "frontier_scores")) {
        has_frontier_scores = true;
        const auto scores = get_values(node.properties, "frontier_scores");
        if (!scores.empty()) {
          const double heading_angle = std::atan2((goal - node_pos).y(), (goal - node_pos).x());
          const double normalized = heading_angle < 0 ? heading_angle + 2 * M_PI : heading_angle;
          const int best_bin = static_cast<int>(std::round(normalized / (2 * M_PI / scores.size()))) % scores.size();
          frontier_score = scores[best_bin];
          cur_frontier_dist_cost_factor = 1.0 - frontier_score_factor_ * std::log(std::max(frontier_score, 1e-6));
        }
      }

      if (!is_scored_graph || !has_frontier_scores) {
        is_scored_graph = false;
        frontier_cost = frontier_path_distance * frontier_dist_cost_factor_;
      } else {
        frontier_cost = frontier_path_distance * cur_frontier_dist_cost_factor;
      }
      frontier_scores_[id] = std::make_pair(node, std::make_pair(frontier_score, frontier_cost));

      if (latest_frontier_) {
        const double distance_to_latest_frontier = (node_pos - *latest_frontier_).norm();
        if (distance_to_latest_frontier < local_frontier_radius_) {
          // Scored graph: require minimum score threshold (matches original WildOS logic).
          // Unscored graph (e.g. mock_nav_graph_publisher, no frontier_scores KeyValue):
          // distance alone is sufficient — frontier_score stays -1.0 so the score
          // check would never pass, breaking local-frontier preference entirely.
          if (!has_frontier_scores || frontier_score > min_local_frontier_score_) {
            local_scored_frontiers.push_back(id);
          }
        }
      }
    }

    const double node_goal_dist = (node_pos - goal).norm();
    if (node_goal_dist < goal_radius) {
      graph[id].push_back({virtual_goal, goal_dist_cost_factor_ * node_goal_dist});
      graph[virtual_goal].push_back({id, goal_dist_cost_factor_ * node_goal_dist});
    }
  }

  bool use_local_frontiers = false;
  if (!local_scored_frontiers.empty()) {
    if (latest_frontier_time_ && (current_time - *latest_frontier_time_).seconds() < path_smoothness_period_) {
      for (const auto id : local_scored_frontiers) {
        const double frontier_cost = frontier_scores_[id].second.second;
        graph[virtual_goal].push_back({id, frontier_cost});
        use_local_frontiers = true;
      }
    } else {
      latest_frontier_time_ = current_time;
    }
  }

  if (!use_local_frontiers) {
    for (const auto& [id, score_pair] : frontier_scores_) {
      graph[virtual_goal].push_back({id, score_pair.second.second});
    }
    latest_frontier_time_ = current_time;
  }

  const auto path_idx = dijkstra_shortest_path(current_node_idx_, virtual_goal, graph);
  std::vector<Eigen::Vector3d> path_points;
  bool has_frontier_in_path = false;
  if (!path_idx.empty()) {
    for (size_t idx = 0; idx < path_idx.size(); ++idx) {
      const auto node_id = path_idx[idx];
      if (node_id >= nodes_.size()) continue;
      const auto& node = nodes_[node_id];
      const auto& pos = node.pose.position;
      path_points.emplace_back(pos.x, pos.y, pos.z);

      if (idx == path_idx.size() - 2 && trav_class_idx_ < node.trav_properties.size() && node.trav_properties[trav_class_idx_].is_frontier) {
        if (!node.trav_properties[trav_class_idx_].frontier_points.empty()) {
          Eigen::Vector3d mean_frontier(0.0, 0.0, 0.0);
          const double npts = static_cast<double>(node.trav_properties[trav_class_idx_].frontier_points.size());
          for (const auto& fp : node.trav_properties[trav_class_idx_].frontier_points) {
            mean_frontier += Eigen::Vector3d(fp.x, fp.y, fp.z);
          }
          mean_frontier /= npts;
          path_points.push_back(mean_frontier);
        }
        latest_frontier_ = Eigen::Vector3d(node.pose.position.x, node.pose.position.y, node.pose.position.z);
        has_frontier_in_path = true;
      }
    }
  }

  if (!has_frontier_in_path) {
    latest_frontier_.reset();
    RCLCPP_WARN(logger_, "NO FRONTIER IN PATH!!!!!");
  }

  return path_points;
}

}  // namespace wildos_graphnav_planner
