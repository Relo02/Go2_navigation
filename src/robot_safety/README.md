# robot_safety

Safety gating layer between Nav2 and `go2_hw_bridge`.

---

## `velocity_limiter` node

Sits between Nav2 `/cmd_vel` and `go2_hw_bridge`.
Subscriptions: `/cmd_vel` (`geometry_msgs/Twist`), `/robot_status` (`robot_common_interfaces/RobotStatus`).
Publication: `/cmd_vel_safe` (`geometry_msgs/Twist`).

**Gating rules (first match wins):**
1. `robot_status.estop_active == true` → publish zero velocity
2. No `/cmd_vel` received within `cmd_timeout` seconds (watchdog) → publish zero velocity
3. Otherwise → forward `/cmd_vel` unchanged

**Integration:** Enable via `go2_bringup.launch.py` with `use_safety:=true`.
The launch file automatically redirects `go2_hw_bridge` to read `/cmd_vel_safe`.

```
Nav2 /cmd_vel  →  velocity_limiter  →  /cmd_vel_safe  →  go2_hw_bridge  →  /api/sport/request
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cmd_timeout` | `0.5` s | Watchdog: publish zero velocity if no `/cmd_vel` within this time |
| `publish_rate` | `20.0` Hz | Output `/cmd_vel_safe` publish rate |
| `cmd_vel_in` | `/cmd_vel` | Input topic (Nav2 velocity commands) |
| `cmd_vel_out` | `/cmd_vel_safe` | Output topic (to `go2_hw_bridge`) |
| `status_topic` | `/robot_status` | `RobotStatus` input topic |

Parameters are set via `safety.launch.py` launch arguments, not `config/safety_params.yaml`.

---

## Launch

| File | Purpose |
|------|---------|
| `launch/safety.launch.py` | Start `velocity_limiter`; included by `go2_bringup.launch.py` |

---

## TODO

- `TODO(safety)`: Implement `tilt_monitor` — IMU-based fall/tilt detection that
  sets `robot_status.estop_active` when tilt exceeds a threshold.
