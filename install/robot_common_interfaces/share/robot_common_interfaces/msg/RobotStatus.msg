# RobotStatus.msg
# Published by each hardware adapter to report robot health.
# Topic: /robot_status  (latched, 1 Hz)
#
# All hardware adapters (go2_bringup, future agibot_d1_bringup) MUST publish this.

std_msgs/Header header

# Battery state of charge [0.0 = empty, 1.0 = full]
float32 battery_soc

# Battery voltage [V]
float32 battery_voltage

# Estop engaged flag
bool estop_active

# Motion mode string (hardware-specific, e.g. "sport", "walk", "idle")
string motion_mode

# True if the hardware interface is alive and responding
bool hw_ok
