"""
Python implementation of graphnav Dijkstra planner.

Mirrors graphnav_planner/src/planner.cpp (nebula2-wildos) in pure Python / numpy.
The planner operates on graphnav_msgs/NavigationGraph messages:

  update_graph(nav_graph_msg)          -- rebuild internal graph from ROS msg
  plan_to_goal(goal_xyz, radius, t)    -- Dijkstra + frontier scoring
                                          returns list of (x,y,z) waypoints

Frontier cost model (same as C++ original):
  frontier_cost = frontier_path_distance * (1 - score_factor * log(score))
  where frontier_path_distance is the BFS distance through unexplored space
  from the frontier point to the goal.

author: Lorenzo Ortolani
"""

from __future__ import annotations

import math
import heapq
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Unexplored-space distance map
# ─────────────────────────────────────────────────────────────────────────────

class UnexploredSpaceMap:
    """
    Binary grid (1=unexplored, 0=explored) with Dijkstra distance-to-goal
    propagated only through unexplored cells.
    Mirrors graphnav_planner::UnexploredSpaceMap.
    """

    _DIRS = [
        ( 1,  0, 1.0),
        ( 0,  1, 1.0),
        (-1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 1,  1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)),
        (-1, -1, math.sqrt(2)),
    ]

    def __init__(
        self,
        min_x: float, max_x: float,
        min_y: float, max_y: float,
        margin: float = 10.0,
        resolution: float = 1.0,
    ):
        self.resolution = resolution
        self.origin_x = min_x - margin
        self.origin_y = min_y - margin
        self.size_x = int(math.ceil((max_x - min_x + 2 * margin) / resolution))
        self.size_y = int(math.ceil((max_y - min_y + 2 * margin) / resolution))
        self.map = np.ones((self.size_x, self.size_y), dtype=np.int8)  # 1=unexplored
        self.dist_map = np.full((self.size_x, self.size_y), np.inf, dtype=np.float32)

    # ------------------------------------------------------------------

    def mark_explored(self, x: float, y: float, radius: float) -> None:
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        ir = int(math.ceil(radius / self.resolution))
        for ddx in range(-ir, ir + 1):
            for ddy in range(-ir, ir + 1):
                if math.hypot(ddx * self.resolution, ddy * self.resolution) <= radius:
                    nx, ny = ix + ddx, iy + ddy
                    if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                        self.map[nx, ny] = 0

    def compute_distance_from(self, x: float, y: float) -> None:
        """BFS/Dijkstra from goal (x,y) through unexplored cells."""
        self.dist_map.fill(np.inf)
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)

        pq: list = []

        if 0 <= ix < self.size_x and 0 <= iy < self.size_y:
            self.dist_map[ix, iy] = 0.0
            heapq.heappush(pq, (0.0, ix, iy))
        else:
            # Goal is outside grid — seed from nearest border cell
            bx = max(0, min(ix, self.size_x - 1))
            by = max(0, min(iy, self.size_y - 1))
            d0 = math.hypot((bx - ix) * self.resolution, (by - iy) * self.resolution)
            self.dist_map[bx, by] = d0
            heapq.heappush(pq, (d0, bx, by))

        while pq:
            cur_dist, cx, cy = heapq.heappop(pq)
            if cur_dist > self.dist_map[cx, cy]:
                continue
            for ddx, ddy, step_cost in self._DIRS:
                nx, ny = cx + ddx, cy + ddy
                if not (0 <= nx < self.size_x and 0 <= ny < self.size_y):
                    continue
                if self.map[nx, ny] == 0:
                    continue  # skip explored cells
                new_dist = self.dist_map[cx, cy] + step_cost * self.resolution
                if new_dist < self.dist_map[nx, ny]:
                    self.dist_map[nx, ny] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

    def query_distance_to(self, x: float, y: float, radius: int = 1) -> float:
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        min_dist = float('inf')
        for ddx in range(-radius, radius + 1):
            for ddy in range(-radius, radius + 1):
                nx, ny = ix + ddx, iy + ddy
                if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                    v = float(self.dist_map[nx, ny])
                    if v < min_dist:
                        min_dist = v
        return min_dist


# ─────────────────────────────────────────────────────────────────────────────
# Main planner
# ─────────────────────────────────────────────────────────────────────────────

class GraphNavPlanner:
    """
    Dijkstra-based global planner on a graphnav_msgs/NavigationGraph.

    Behaviour is identical to graphnav_planner::Planner (C++) including:
      - traversability-class filtering on edges/nodes
      - unexplored-space distance computation
      - visual frontier scoring via angular bins
      - local-frontier exploitation with path_smoothness_period
      - virtual goal node connected to any graph node within goal_radius
    """

    # Internal sentinel index for the virtual goal node
    _VGOAL = -1

    def __init__(self, logger=None):
        self._log = logger

        # ── Tunable parameters (mirrors PlannerNode declare_parameter) ──
        self.frontier_dist_cost_factor: float = 2.0
        self.goal_dist_cost_factor: float     = 1.0
        self.frontier_score_factor: float     = 10.0
        self.min_local_frontier_score: float  = 0.4
        self.local_frontier_radius: float     = 7.0
        self.path_smoothness_period: float    = 10.0  # seconds

        # ── Internal state ───────────────────────────────────────────────
        self._trav_class: str = 'default'
        self._trav_class_idx: int = 0
        self._current_node_idx: int = 0

        # node_idx -> graphnav_msgs/Node
        self._nodes: dict = {}
        # node_idx -> [(neighbor_idx, edge_weight)]
        self._adj: dict = {}

        self._unexplored_map: Optional[UnexploredSpaceMap] = None
        self._latest_frontier: Optional[tuple] = None        # (x,y,z)
        self._latest_frontier_time: Optional[float] = None  # seconds

        # idx -> (node_msg, score: float, cost: float)
        self._frontier_scores: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_trav_class(self, trav_class: str) -> None:
        self._trav_class = trav_class

    def update_graph(self, nav_graph_msg) -> None:
        """Parse a graphnav_msgs/NavigationGraph message."""
        self._nodes = {}
        self._adj = {}

        for i, node in enumerate(nav_graph_msg.nodes):
            self._nodes[i] = node
            self._adj[i] = []

        trav_classes = list(nav_graph_msg.trav_classes)
        if self._trav_class in trav_classes:
            self._trav_class_idx = trav_classes.index(self._trav_class)
        else:
            if self._log:
                self._log.warning(
                    f'Traversability class "{self._trav_class}" not found in graph, using index 0'
                )
            self._trav_class_idx = 0

        for edge in nav_graph_msg.edges:
            fi, ti = int(edge.from_idx), int(edge.to_idx)
            if fi not in self._adj:
                self._adj[fi] = []
            if ti not in self._adj:
                self._adj[ti] = []
            if self._trav_class_idx < len(edge.traversability):
                w = float(edge.traversability[self._trav_class_idx].traversability_cost)
            else:
                w = 1.0
            self._adj[fi].append((ti, w))
            self._adj[ti].append((fi, w))

        self._current_node_idx = int(nav_graph_msg.current_node_idx)
        self._unexplored_map = self._build_unexplored_map()

    def plan_to_goal(
        self,
        goal_xyz: tuple,
        goal_radius: float,
        current_time_sec: float,
    ) -> list:
        """
        Plan from current_node_idx to goal_xyz using Dijkstra + frontier scoring.

        Parameters
        ----------
        goal_xyz        : (x, y, z) goal position in graph frame
        goal_radius     : nodes within this distance are connected to virtual goal
        current_time_sec: wall/sim time in seconds (for path_smoothness_period)

        Returns
        -------
        List of (x, y, z) waypoints from robot to goal (empty if no path found).
        """
        if self._unexplored_map is None or not self._nodes:
            return []

        gx, gy, gz = float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])
        self._unexplored_map.compute_distance_from(gx, gy)

        VGOAL = self._VGOAL

        # Build local adjacency (copy + virtual-goal edges)
        adj: dict = {k: list(v) for k, v in self._adj.items()}
        adj[VGOAL] = []

        self._frontier_scores = {}
        local_scored_frontiers: list = []
        is_scored_graph = True
        goal_vec = np.array([gx, gy, gz])

        for idx, node in self._nodes.items():
            nx = float(node.pose.position.x)
            ny = float(node.pose.position.y)
            nz = float(node.pose.position.z)
            node_xyz = np.array([nx, ny, nz])

            # Connect node to virtual goal if close enough
            node_goal_dist = float(np.linalg.norm(node_xyz - goal_vec))
            if node_goal_dist < goal_radius:
                goal_cost = self.goal_dist_cost_factor * node_goal_dist
                adj[idx].append((VGOAL, goal_cost))

            # Frontier handling
            tp = node.trav_properties
            if self._trav_class_idx >= len(tp) or not tp[self._trav_class_idx].is_frontier:
                continue

            tpk = tp[self._trav_class_idx]

            # Distance through unexplored space to the goal
            frontier_path_distance = float('inf')
            for fp in tpk.frontier_points:
                d = self._unexplored_map.query_distance_to(float(fp.x), float(fp.y))
                if d < frontier_path_distance:
                    frontier_path_distance = d

            # Heading from this node toward the goal
            heading = goal_vec - node_xyz
            heading_norm = np.linalg.norm(heading[:2])
            if heading_norm > 1e-6:
                heading = heading / heading_norm

            frontier_score = -1.0
            has_frontier_scores = False
            cur_fdcf = self.frontier_dist_cost_factor

            for kv in node.properties:
                if kv.key == 'frontier_scores':
                    has_frontier_scores = True
                    num_bins = len(kv.value)
                    if num_bins > 0:
                        angle_per_bin = 2.0 * math.pi / num_bins
                        ha = math.atan2(heading[1], heading[0])
                        if ha < 0:
                            ha += 2.0 * math.pi
                        best_bin = int(round(ha / angle_per_bin)) % num_bins
                        frontier_score = float(kv.value[best_bin])
                        if frontier_score > 1e-9:
                            cur_fdcf = 1.0 - self.frontier_score_factor * math.log(frontier_score)

                    # Local-frontier tracking
                    if self._latest_frontier is not None:
                        lf = np.array(self._latest_frontier)
                        if (np.linalg.norm(node_xyz - lf) < self.local_frontier_radius
                                and frontier_score > self.min_local_frontier_score):
                            local_scored_frontiers.append(idx)
                    break

            if not has_frontier_scores:
                is_scored_graph = False
                frontier_cost = frontier_path_distance * self.frontier_dist_cost_factor
                if self._latest_frontier is not None:
                    lf = np.array(self._latest_frontier)
                    if np.linalg.norm(node_xyz - lf) < self.local_frontier_radius:
                        local_scored_frontiers.append(idx)
            else:
                frontier_cost = frontier_path_distance * cur_fdcf

            self._frontier_scores[idx] = (node, frontier_score, frontier_cost)

        # ── Local-frontier exploitation ──────────────────────────────
        use_local = False
        if local_scored_frontiers:
            dt = (
                current_time_sec - self._latest_frontier_time
                if self._latest_frontier_time is not None
                else float('inf')
            )
            if dt < self.path_smoothness_period:
                for idx in local_scored_frontiers:
                    fc = self._frontier_scores[idx][2]
                    adj[idx].append((VGOAL, fc))
                use_local = True
            else:
                self._latest_frontier_time = current_time_sec

        if not use_local:
            for idx, (_, _, fc) in self._frontier_scores.items():
                adj[idx].append((VGOAL, fc))
            self._latest_frontier_time = current_time_sec

        # ── Dijkstra ─────────────────────────────────────────────────
        dist: dict = {k: float('inf') for k in list(self._nodes.keys()) + [VGOAL]}
        dist[self._current_node_idx] = 0.0
        prev: dict = {}
        pq = [(0.0, self._current_node_idx)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist[VGOAL] == float('inf'):
            if self._log:
                self._log.warning('GraphNavPlanner: no path to goal found')
            return []

        # ── Reconstruct path indices ─────────────────────────────────
        path_indices = []
        cur = VGOAL
        while cur in prev:
            path_indices.append(cur)
            cur = prev[cur]
        path_indices.append(cur)
        path_indices.reverse()

        # ── Convert to world coords, handle frontier tip ─────────────
        path_points = []
        has_frontier = False

        for i, idx in enumerate(path_indices):
            if idx == VGOAL:
                continue
            node = self._nodes[idx]
            path_points.append((
                float(node.pose.position.x),
                float(node.pose.position.y),
                float(node.pose.position.z),
            ))

            # Second-to-last in the non-virtual sequence: check for frontier tip
            is_second_to_last = (
                i == len(path_indices) - 2
                or (i == len(path_indices) - 1 and path_indices[-1] == VGOAL)
            )
            if not is_second_to_last:
                continue
            tp = node.trav_properties
            if self._trav_class_idx >= len(tp) or not tp[self._trav_class_idx].is_frontier:
                continue
            tpk = tp[self._trav_class_idx]
            if tpk.frontier_points:
                xs = [float(fp.x) for fp in tpk.frontier_points]
                ys = [float(fp.y) for fp in tpk.frontier_points]
                zs = [float(fp.z) for fp in tpk.frontier_points]
                path_points.append((
                    sum(xs) / len(xs),
                    sum(ys) / len(ys),
                    sum(zs) / len(zs),
                ))
            self._latest_frontier = (
                float(node.pose.position.x),
                float(node.pose.position.y),
                float(node.pose.position.z),
            )
            has_frontier = True

        if not has_frontier:
            self._latest_frontier = None

        return path_points

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_unexplored_map(self) -> Optional[UnexploredSpaceMap]:
        if not self._nodes:
            return None
        xs = [float(n.pose.position.x) for n in self._nodes.values()]
        ys = [float(n.pose.position.y) for n in self._nodes.values()]
        umap = UnexploredSpaceMap(min(xs), max(xs), min(ys), max(ys))
        for node in self._nodes.values():
            tp = node.trav_properties
            if self._trav_class_idx < len(tp):
                r = float(tp[self._trav_class_idx].explored_radius)
                if r > 0:
                    umap.mark_explored(
                        float(node.pose.position.x),
                        float(node.pose.position.y),
                        r,
                    )
        return umap
