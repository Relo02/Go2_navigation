"""
CasADi / IPOPT MPC trajectory tracker for Go2 quadruped (2-D kinematic model).

Design
------
- Tracks a path produced by AStarPlanner for ground-based motion.
  Path waypoints are 2-D (x, y).

- Uses a 2-D kinematic model (no z optimization):
    State    x = [px, py, yaw]              (NX = 3)
    Control  u = [vx, vy, omega]            (NU = 3)
    
  Dynamics (Euler integration):
    px_{k+1}  = px_k + (vx_k*cos(yaw_k) - vy_k*sin(yaw_k))*dt
    py_{k+1}  = py_k + (vx_k*sin(yaw_k) + vy_k*cos(yaw_k))*dt
    yaw_{k+1} = yaw_k + omega_k*dt

- Obstacle avoidance uses a logistic (sigmoid) barrier applied to predicted
  xy positions over the full horizon:

      J_obs += w_c / (1 + exp( alpha * (dist(X_k, p_obs) - r) ))

  Reference: "UAV trajectory optimisation with MPC" arxiv 2410.09799.

- The NLP is built ONCE with CasADi Opti parameters for x0, x_ref, and
  obstacle positions (padded to a fixed count with far sentinels).
  Only numeric values are updated each solve call — no symbolic rebuilding.

Dependencies: casadi, numpy
author: Lorenzo Ortolani (adapted for Go2 quadruped by user)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import casadi as ca

from graphnav_mpc_planner.gaussian_grid_map import FixedGaussianGridMap


# ============================================================
# Configuration
# ============================================================

@dataclass
class MPCConfig:
    """All tunable MPC parameters for 2-D kinematic motion model."""

    # Horizon
    N: int = 30
    dt: float = 0.1

    # Velocity limits (kinematic model)
    vx_max: float = 1.0           # max forward velocity [m/s]
    vy_max: float = 0.5           # max lateral velocity [m/s]
    omega_max: float = 1.5        # max angular velocity [rad/s]

    # Desired cruise speed (forward direction)
    v_ref: float = 0.5             # [m/s]

    # Tracking cost weights
    Q_xy: float = 20.0             # position tracking
    Q_yaw: float = 0.5            # yaw tracking
    Q_terminal: float = 50.0       # terminal state weight multiplier
    R_vel: float = 1.0             # velocity control effort
    R_omega: float = 0.5          # angular velocity control effort
    R_jerk: float = 0.2           # smoothness (change in velocity)

    # Logistic sigmoid obstacle barrier
    W_obs_sigmoid: float = 500.0   # weight per obstacle point per step
    obs_alpha: float = 8.0         # logistic steepness [1/m]
    obs_r: float = 0.8            # safety radius [m]

    # LiDAR point selection
    max_obs_constraints: int = 15  # points used per solve (padded with sentinels)
    obs_check_radius: float = 3.0  # only consider points within this radius [m]

    # IPOPT
    max_iter: int = 100
    warm_start: bool = True
    print_level: int = 0


# ============================================================
# Result
# ============================================================

@dataclass
class MPCResult:
    success: bool
    x_pred: np.ndarray              # (N+1, 3)  [px, py, yaw]
    u_opt: np.ndarray               # (N,   3)  [vx, vy, omega]
    cost: float
    solve_time_ms: float
    security_mode: bool = False     # True when security escape protocol is active

    @property
    def next_position(self) -> np.ndarray:
        """Next predicted (px, py) position."""
        return self.x_pred[1, :2]

    @property
    def next_yaw(self) -> float:
        """Next predicted yaw [rad]."""
        return float(self.x_pred[1, 2])

    @property
    def predicted_xy(self) -> np.ndarray:
        """All predicted (px, py) positions, shape (N+1, 2)."""
        return self.x_pred[:, :2]

    @property
    def predicted_yaw(self) -> np.ndarray:
        """All predicted yaw values, shape (N+1,)."""
        return self.x_pred[:, 2]


# ============================================================
# MPC Tracker
# ============================================================

class MPCTracker:
    """
    2-D path-tracking MPC with logistic sigmoid obstacle barrier for quadrupeds.

    State:   [px, py, yaw]
    Control: [vx, vy, omega]
    
    Typical usage
    -------------
    tracker = MPCTracker(config)
    result  = tracker.solve(robot_state, a_star_path, obstacle_points_2d=lidar_xy)
    # use result.next_position as velocity setpoint
    """

    NX = 3   # [px, py, yaw]
    NU = 3   # [vx, vy, omega]
    _OBS_SENTINEL = 1e3

    def __init__(self, config: Optional[MPCConfig] = None):
        self.cfg = config or MPCConfig()

        # Warm-start storage
        self._prev_u: Optional[np.ndarray] = None   # (N, NU)
        self._prev_x: Optional[np.ndarray] = None   # (N+1, NX)

        # Cached parametric NLP — built once (n_obs is always max_obs_constraints)
        self._nlp_built: bool = False
        self._opti:   Optional[ca.Opti] = None
        self._X:      Optional[ca.MX]   = None
        self._U:      Optional[ca.MX]   = None
        self._p_x0:   Optional[ca.MX]   = None
        self._p_xref: Optional[ca.MX]   = None
        self._p_obs:  Optional[ca.MX]   = None  # (2, max_obs_constraints)

        # Forward-only path progress
        self._path_progress_idx: int = 0
        self._last_valid_x0: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Grid map (API compatibility with mpc_node.py — not used in NLP)
    # ------------------------------------------------------------------

    def update_grid(self, grid_map: FixedGaussianGridMap) -> None:
        """No-op: obstacle avoidance uses direct LiDAR point barriers."""
        pass

    # ------------------------------------------------------------------
    # LiDAR point selection
    # ------------------------------------------------------------------

    def _select_obs_points(
        self,
        pts_2d: np.ndarray,
        drone_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Return up to max_obs_constraints points within obs_check_radius,
        nearest-first.  Always padded to exactly max_obs_constraints rows
        with far sentinels (1e4, 1e4) so the NLP structure never changes.
        """
        n_target = self.cfg.max_obs_constraints

        if len(pts_2d) > 0:
            # Keep only finite obstacle points to protect the NLP parameters.
            finite_mask = np.isfinite(pts_2d).all(axis=1)
            pts_2d = pts_2d[finite_mask]
        if len(pts_2d) > 0:
            dists = np.linalg.norm(pts_2d - drone_xy, axis=1)
            mask  = dists < self.cfg.obs_check_radius
            if np.any(mask):
                close = pts_2d[mask]
                d_close = dists[mask]
                n_sel = min(len(close), n_target)
                idx   = np.argsort(d_close)[:n_sel]
                selected = close[idx]
            else:
                selected = np.empty((0, 2))
        else:
            selected = np.empty((0, 2))

        n_found = len(selected)
        if n_found < n_target:
            sentinel = np.full((n_target - n_found, 2), self._OBS_SENTINEL)
            selected = np.vstack([selected, sentinel]) if n_found > 0 else sentinel

        return selected   # (max_obs_constraints, 2)

    # ------------------------------------------------------------------
    # Parametric NLP — built once
    # ------------------------------------------------------------------

    def _build_nlp(self) -> None:
        """
        Build the parametric NLP for 2-D kinematic motion model.

        State: [px, py, yaw]
        Control: [vx, vy, omega]
        
        Dynamics:
            px_{k+1}  = px_k + (vx_k*cos(yaw_k) - vy_k*sin(yaw_k))*dt
            py_{k+1}  = py_k + (vx_k*sin(yaw_k) + vy_k*cos(yaw_k))*dt
            yaw_{k+1} = yaw_k + omega_k*dt
        """
        cfg = self.cfg
        N, dt = cfg.N, cfg.dt
        NX, NU = self.NX, self.NU
        n_obs = cfg.max_obs_constraints

        opti   = ca.Opti()
        X      = opti.variable(NX, N + 1)
        U      = opti.variable(NU, N)
        p_x0   = opti.parameter(NX)
        p_xref = opti.parameter(NX, N + 1)
        p_obs  = opti.parameter(2, n_obs)     # always present (sentinels when sparse)

        # ── Weight matrices ───────────────────────────────────────────
        q = np.array([
            cfg.Q_xy,   # px weight
            cfg.Q_xy,   # py weight
            cfg.Q_yaw,  # yaw weight
        ])
        Q   = np.diag(q)
        Q_T = np.diag(q * cfg.Q_terminal)
        R   = np.diag([cfg.R_vel, cfg.R_vel, cfg.R_omega])

        # ── Objective ────────────────────────────────────────────────
        cost = 0.0

        for k in range(N):
            # State tracking
            e = X[:, k] - p_xref[:, k]
            cost += ca.mtimes([e.T, Q, e])

            # Control effort
            u_k = U[:, k]
            cost += ca.mtimes([u_k.T, R, u_k])

            # Jerk smoothness (change in velocity)
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                cost += cfg.R_jerk * ca.dot(du, du)

            # Logistic sigmoid barrier (all steps including k=0)
            for j in range(n_obs):
                dist_k = ca.sqrt(
                    (X[0, k] - p_obs[0, j]) ** 2 +
                    (X[1, k] - p_obs[1, j]) ** 2 + 1e-6
                )
                # Numerically stable logistic barrier:
                # 1/(1+exp(s)) == 0.5 * (1 - tanh(s/2))
                s_k = cfg.obs_alpha * (dist_k - cfg.obs_r)
                cost += cfg.W_obs_sigmoid * 0.5 * (1.0 - ca.tanh(0.5 * s_k))

        # Terminal cost
        e_T = X[:, N] - p_xref[:, N]
        cost += ca.mtimes([e_T.T, Q_T, e_T])

        for j in range(n_obs):
            dist_T = ca.sqrt(
                (X[0, N] - p_obs[0, j]) ** 2 +
                (X[1, N] - p_obs[1, j]) ** 2 + 1e-6
            )
            s_T = cfg.obs_alpha * (dist_T - cfg.obs_r)
            cost += cfg.W_obs_sigmoid * 0.5 * (1.0 - ca.tanh(0.5 * s_T))

        opti.minimize(cost)

        # ── Dynamics (2-D kinematic, Euler) ──────────────────────────
        for k in range(N):
            px_k   = X[0, k]
            py_k   = X[1, k]
            yaw_k  = X[2, k]
            vx_k   = U[0, k]
            vy_k   = U[1, k]
            omega_k = U[2, k]
            
            # Kinematic bicycle model with lateral velocity
            cos_yaw = ca.cos(yaw_k)
            sin_yaw = ca.sin(yaw_k)
            
            opti.subject_to(X[0, k+1] == px_k + (vx_k*cos_yaw - vy_k*sin_yaw)*dt)
            opti.subject_to(X[1, k+1] == py_k + (vx_k*sin_yaw + vy_k*cos_yaw)*dt)
            opti.subject_to(X[2, k+1] == yaw_k + omega_k*dt)

        opti.subject_to(X[:, 0] == p_x0)

        # ── Box constraints ───────────────────────────────────────────
        for k in range(N):
            opti.subject_to(opti.bounded(-cfg.vx_max,     U[0, k],  cfg.vx_max))
            opti.subject_to(opti.bounded(-cfg.vy_max,     U[1, k],  cfg.vy_max))
            opti.subject_to(opti.bounded(-cfg.omega_max,  U[2, k],  cfg.omega_max))

        # ── Solver ────────────────────────────────────────────────────
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {
            'max_iter':               cfg.max_iter,
            'print_level':            cfg.print_level,
            'sb':                     'yes',
            'warm_start_init_point':  'yes' if cfg.warm_start else 'no',
        }
        opti.solver('ipopt', p_opts, s_opts)

        self._opti      = opti
        self._X         = X
        self._U         = U
        self._p_x0      = p_x0
        self._p_xref    = p_xref
        self._p_obs     = p_obs
        self._nlp_built = True

        # Invalidate warm start after rebuild
        self._prev_u = None
        self._prev_x = None

    # ------------------------------------------------------------------
    # Reference trajectory
    # ------------------------------------------------------------------

    def _build_reference(
        self,
        robot_state: np.ndarray,
        path_world: list,
    ) -> np.ndarray:
        """
        Build an (N+1, NX) reference trajectory by advancing along the A* path
        at v_ref m/s from the closest forward waypoint.
        
        Implements TRUE ONLINE REPLANNING by:
        1. Always starting from CURRENT robot position
        2. Finding closest path point considering forward progress
        3. Continuously re-anchoring to actual robot pose instead of maintaining
           a fixed trajectory index across MPC solves
        
        Parameters
        ----------
        robot_state : (3,) [px, py, yaw]
        path_world  : list of (x, y) waypoints from A*
        
        Returns
        -------
        x_ref : (N+1, 3) reference trajectory [px, py, yaw]
        """
        N, dt, v_ref = self.cfg.N, self.cfg.dt, self.cfg.v_ref
        x_ref = np.zeros((N + 1, self.NX))

        if not path_world or len(path_world) < 2:
            for k in range(N + 1):
                x_ref[k] = robot_state
            return x_ref

        path    = np.array(path_world, dtype=float)[:, :2]  # only use x, y
        path_xy = path
        
        diffs_xy  = np.diff(path_xy, axis=0)
        seg_len   = np.hypot(diffs_xy[:, 0], diffs_xy[:, 1])
        arc       = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_arc = arc[-1]

        # === KEY FIX: Find closest waypoint using forward-aware search ===
        # This ensures true online replanning by always finding where the
        # robot ACTUALLY is on the path, not maintaining a stale index
        robot_xy  = robot_state[:2]
        distances = np.linalg.norm(path_xy - robot_xy, axis=1)
        i_closest = int(np.argmin(distances))
        
        # Sanity check: if closest point is very far away, robot may have drifted
        # from path (e.g., during obstacle avoidance). This is expected and OK.
        # The MPC will naturally steer back toward the path.
        closest_distance = distances[i_closest]
        
        s0 = arc[i_closest]  # Arc length at closest point

        # IMPORTANT: First reference point should match robot's CURRENT state exactly
        # This ensures smooth online replanning without discontinuities
        x_ref[0, 0] = robot_state[0]
        x_ref[0, 1] = robot_state[1]
        x_ref[0, 2] = robot_state[2]

        for k in range(1, N + 1):
            s_k = min(s0 + v_ref * k * dt, total_arc)
            idx = int(np.searchsorted(arc, s_k, side='right')) - 1
            idx = np.clip(idx, 0, len(path_xy) - 2)

            seg_l   = seg_len[idx]
            t       = np.clip((s_k - arc[idx]) / (seg_l + 1e-9), 0.0, 1.0)
            pos_xy  = path_xy[idx] + t * diffs_xy[idx]
            seg_dir = diffs_xy[idx] / (seg_l + 1e-9)
            yaw_k   = np.arctan2(seg_dir[1], seg_dir[0])

            x_ref[k, 0] = pos_xy[0]
            x_ref[k, 1] = pos_xy[1]
            x_ref[k, 2] = yaw_k

        return x_ref

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        robot_state: np.ndarray,
        path_world: list,
        obstacle_points_2d: Optional[np.ndarray] = None,
    ) -> MPCResult:
        """
        Solve the MPC optimization for 2-D motion.

        Parameters
        ----------
        robot_state        : (3,) [px, py, yaw]
        path_world         : list of (x, y) waypoints from A*
        obstacle_points_2d : (M, 2) raw LiDAR obstacle positions in world x-y
        """
        t0  = time.perf_counter()
        cfg = self.cfg
        N   = cfg.N
        NX, NU = self.NX, self.NU

        x0 = np.asarray(robot_state, dtype=float)
        if len(x0) != NX:
            raise ValueError(f"Expected state of length {NX}, got {len(x0)}")
        if not np.isfinite(x0).all():
            if self._last_valid_x0 is not None and np.isfinite(self._last_valid_x0).all():
                x0 = self._last_valid_x0.copy()
            else:
                x0 = np.zeros((NX,), dtype=float)
        self._last_valid_x0 = x0.copy()

        # Clamp forward-progress index to current path length
        path_len = len(path_world) if path_world else 0
        self._path_progress_idx = min(self._path_progress_idx, max(path_len - 1, 0))

        x_ref = self._build_reference(x0, path_world)
        if not np.isfinite(x_ref).all():
            x_ref = np.repeat(x0[None, :], N + 1, axis=0)

        # Build obstacle point array (always max_obs_constraints rows)
        robot_xy = x0[:2]
        if obstacle_points_2d is not None and len(obstacle_points_2d) > 0:
            obs_pts = self._select_obs_points(obstacle_points_2d, robot_xy)
        else:
            obs_pts = np.full((cfg.max_obs_constraints, 2), self._OBS_SENTINEL)
        if not np.isfinite(obs_pts).all():
            obs_pts = np.full((cfg.max_obs_constraints, 2), self._OBS_SENTINEL)

        # Build NLP once
        if not self._nlp_built:
            self._build_nlp()

        opti = self._opti

        # ── Update parameter values ──────────────────────────────────
        opti.set_value(self._p_x0,   x0)
        opti.set_value(self._p_xref, x_ref.T)       # (NX, N+1)
        opti.set_value(self._p_obs,  obs_pts.T)      # (2, n_obs)

        # ── Warm start ───────────────────────────────────────────────
        if cfg.warm_start and self._prev_u is not None:
            try:
                opti.set_initial(self._U, self._prev_u.T)
                opti.set_initial(self._X, self._prev_x.T)
            except Exception:
                opti.set_initial(self._X, x_ref.T)
                opti.set_initial(self._U, np.zeros((NU, N)))
        else:
            opti.set_initial(self._X, x_ref.T)
            opti.set_initial(self._U, np.zeros((NU, N)))

        # ── Solve ────────────────────────────────────────────────────
        try:
            sol      = opti.solve()
            success  = True
            cost_val = float(sol.value(opti.f))
        except RuntimeError:
            sol      = opti.debug
            success  = False
            try:
                cost_val = float(sol.value(opti.f))
            except Exception:
                cost_val = float('inf')

        # ── Extract solution ─────────────────────────────────────────
        try:
            U_opt = np.array(sol.value(self._U), dtype=float)
            X_opt = np.array(sol.value(self._X), dtype=float)
            if np.any(np.isnan(U_opt)) or np.any(np.isnan(X_opt)):
                raise ValueError('NaN in solution')
            u_seq  = U_opt.T    # (N,  NU)
            x_pred = X_opt.T    # (N+1, NX)
            self._prev_u = np.vstack([u_seq[1:],  u_seq[-1:]])
            self._prev_x = np.vstack([x_pred[1:], x_pred[-1:]])
        except Exception:
            success = False
            prev_u = self._prev_u
            prev_x = self._prev_x
            # Reset warm-start on failed/invalid solve to avoid poisoning future iterations.
            self._prev_u = None
            self._prev_x = None
            if prev_u is not None and prev_x is not None:
                u_seq  = np.vstack([prev_u[1:], prev_u[-1:]])
                x_pred = np.vstack([prev_x[1:], prev_x[-1:]])
            else:
                u_seq  = np.zeros((N, NU))
                x_pred = x_ref.copy()

        return MPCResult(
            success=success,
            x_pred=x_pred,
            u_opt=u_seq,
            cost=cost_val,
            solve_time_ms=(time.perf_counter() - t0) * 1e3,
        )
