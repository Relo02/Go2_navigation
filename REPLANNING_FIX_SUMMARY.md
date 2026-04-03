# A* MPC Planner - Online Replanning Issue Fix

## Problem Summary

When you set a goal in RViz, the planner initially creates a path, but as the robot moves, the planner doesn't continuously adapt based on the robot's actual pose and changing sensor data. Instead, it appears to just follow the initially planned path and only re-adapt if obstacles directly block the current path.

### Why This Happens

The issue has **three root causes**:

#### 1. **A* Replanning Rate Too Slow (2.5 Hz)**
- **Old Config**: `replan_rate_hz: 2.5` = plans every 400ms
- **MPC Rate**: `mpc_rate_hz: 10.0` = optimizes every 100ms
- **Problem**: For a robot moving at 0.5 m/s, it travels **0.2m per A* cycle**. Meanwhile, the MPC solves 4 times using the same stale path before A* replans again.
- **Effect**: The LiDAR scans shift and new obstacles appear/disappear, but the planning doesn't adapt frequently enough.

#### 2. **MPC Using Stale Path Data**
- A* publishes at 2.5 Hz, but MPC runs at 10 Hz
- MPC solves 3-4 times with the same A* path before receiving a new one
- The reference trajectory building didn't explicitly ensure the trajectory started from the robot's ACTUAL current position

#### 3. **Setpoint Filtering Lag**
- Even when new paths arrived, the old setpoint filtering continued along stale trajectories
- No explicit reset when a fresh path was received

---

## Fixes Applied

### Fix 1: Increased A* Planning Rate to 10 Hz
**File**: `a_star_mpc_planner/config/planner_params.yaml`

```yaml
# BEFORE:
replan_rate_hz: 2.5

# AFTER:
replan_rate_hz: 10.0  # Matches MPC frequency for continuous adaptation
```

**Benefit**: A* now replans every 100ms (instead of every 400ms), ensuring the planning continuously adapts to new robot poses and sensor data.

---

### Fix 2: Improved MPC Reference Trajectory Building
**File**: `a_star_mpc_planner/a_star_mpc_planner/mpc_tracker.py`

#### What Changed
The `_build_reference()` method now implements TRUE ONLINE REPLANNING:

1. **Always starts from actual robot position** (instead of interpolating from stale trajectory index):
   ```python
   # First reference point matches robot's CURRENT state exactly
   x_ref[0, 0] = robot_state[0]    # px
   x_ref[0, 1] = robot_state[1]    # py
   x_ref[0, 2] = robot_state[2]    # yaw
   ```

2. **Re-finds closest waypoint at each MPC solve** (not maintaining a stale index):
   ```python
   # Find where robot ACTUALLY is on the path right now
   distances = np.linalg.norm(path_xy - robot_xy, axis=1)
   i_closest = int(np.argmin(distances))
   s0 = arc[i_closest]
   ```

3. **Builds fresh trajectory from current position** at every 100ms MPC solve

**Benefit**: The MPC always knows where the robot actually is on the path and builds a fresh, updated trajectory rather than extrapolating from a stale trajectory.

---

### Fix 3: Reset Setpoint Filter on New Path
**File**: `a_star_mpc_planner/a_star_mpc_planner/mpc_node.py`

```python
# When new A* path arrives, RESET the setpoint filter
self._setpoint_filtered_xy = None
self._setpoint_filtered_yaw = None
```

**Benefit**: Removes lag from old setpoint filtering, allowing immediate response to new path data.

---

### Fix 4: Enhanced Logging
**Files**: 
- `a_star_node.py` - Added detailed comments explaining continuous replanning
- `mpc_node.py` - Now logs robot position and path update info

**Benefit**: You can now verify that continuous adaptation is happening by watching the logs.

---

## How It Works Now

### Before (Old Behavior)
```
Time 0ms:   [A* replans → new path published]
Time 100ms: [MPC solve 1 - uses path from 0ms]
Time 200ms: [MPC solve 2 - uses path from 0ms]
Time 300ms: [MPC solve 3 - uses path from 0ms]
Time 400ms: [MPC solve 4 - uses path from 0ms]
            [A* replans → new path published] ← Takes 400ms to adapt!
```

### After (New Behavior)
```
Time 0ms:   [A* replan-1 → path-1 published]
Time 100ms: [A* replan-2 → path-2 published] [MPC solve 1 w/ path-2]
Time 200ms: [A* replan-3 → path-3 published] [MPC solve 2 w/ path-3]
Time 300ms: [A* replan-4 → path-4 published] [MPC solve 3 w/ path-4]
Time 400ms: [A* replan-5 → path-5 published] [MPC solve 4 w/ path-5]
            ↑ Planning adapts every 100ms instead of every 400ms!
```

---

## How to Verify the Fix

### 1. Launch the Simulation
```bash
ros2 launch robot_sim sim_a_star_mpc.launch.py
```

### 2. Monitor the Log Output
Look for these patterns showing continuous replanning:

```
[a_star_node] [A*] path=XX wpts local_goal=... dist_global=X.XX m
              ↑ Should see this every 100ms

[mpc_node] [MPC] #XXXX ok=1 cost=XXX solve=XX.X ms
           robot=[1.23, 4.56] setpt=[1.45, 4.78]
           ↑ Robot position should be constantly updating
```

### 3. Set a Goal in RViz (2D Goal Pose tool)
- Place goal on one side of the map
- As robot moves toward it, observe:
  - LiDAR scans shift with robot movement
  - Robot path continuously adapts (not following initial plan rigidly)
  - Setpoint changes smoothly based on new sensor data
  - Log output shows new paths arriving frequently

### 4. Create a Dynamic Obstacle
- If using Gazebo, move obstacles during execution
- Robot should re-plan around them within ~100-200ms (1-2 MPC cycles)
- Previously would take ~400-500ms

---

## Technical Details

### A* Node Online Replanning
At each 100ms tick:
1. **Gets latest robot pose** from `/go2/pose`
2. **Gets latest LiDAR scan** from `/lidar/points_filtered`
3. **Re-centers occupancy grid** on robot's current position
4. **Runs A* search** from robot's current position to goal
5. **Publishes new path**

This is already happening in the code, but now it happens 4x more frequently (10 Hz vs 2.5 Hz).

### MPC Node Online Feedback Control
At each 100ms tick:
1. **Receives latest A* path** via subscription callback
2. **Gets current robot state** from `/go2/pose`
3. **Builds reference trajectory** by:
   - Finding where robot actually is on the path
   - Interpolating ahead at v_ref speed
   - Ensuring trajectory starts from current position
4. **Solves MPC optimization** with:
   - Current state as initial condition
   - Fresh reference trajectory
   - Latest LiDAR obstacles
5. **Publishes setpoint**

Now the reference trajectory is built fresh every 100ms from the actual robot position, not extrapolated from a stale index.

---

## Expected Improvements

✅ **Faster Path Adaptation**: Planner adapts within ~100-200ms of pose change  
✅ **Better Obstacle Avoidance**: Reacts to new obstacles in real-time  
✅ **Smoother Trajectory**: Continuous replanning prevents path jumping  
✅ **More Responsive to Sensor Changes**: LiDAR updates immediately influence planning  
✅ **Actual Online Feedback Control**: MPC truly operates as an online controller

---

## If You Still See Issues

### Slow adaptation (>500ms response time):
- Check A* log output — should see new paths every 100ms
- Verify `replan_rate_hz: 10.0` in config file
- Check that A* doesn't time out (`plan()` should complete in <100ms)

### Path oscillation:
- Increase `mpc_setpoint_alpha: 0.35` (more filtering)
- Decrease `mpc_lookahead_dist` (shorter lookahead)

### MPC solver failures:
- Decrease `mpc_W_obs_sigmoid` (weaker obstacle penalty)
- Increase `mpc_max_iter` (more optimization iterations)

---

## Summary

The issue was that the planner updated too slowly (2.5 Hz A* vs 10 Hz MPC) and didn't fully embrace online replanning principles. The fixes increase A* to 10 Hz and ensure MPC always builds fresh reference trajectories from the robot's actual current position. Now the planner truly adapts as the robot moves and sensors update, rather than following a stale initial plan.
