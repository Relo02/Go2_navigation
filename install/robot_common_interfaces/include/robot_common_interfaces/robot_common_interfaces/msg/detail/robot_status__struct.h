// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_common_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

#ifndef ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_
#define ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'motion_mode'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/RobotStatus in the package robot_common_interfaces.
/**
  * RobotStatus.msg
  * Published by each hardware adapter to report robot health.
  * Topic: /robot_status  (latched, 1 Hz)
  *
  * All hardware adapters (go2_bringup, future agibot_d1_bringup) MUST publish this.
 */
typedef struct robot_common_interfaces__msg__RobotStatus
{
  std_msgs__msg__Header header;
  /// Battery state of charge [0.0 = empty, 1.0 = full]
  float battery_soc;
  /// Battery voltage
  float battery_voltage;
  /// Estop engaged flag
  bool estop_active;
  /// Motion mode string (hardware-specific, e.g. "sport", "walk", "idle")
  rosidl_runtime_c__String motion_mode;
  /// True if the hardware interface is alive and responding
  bool hw_ok;
} robot_common_interfaces__msg__RobotStatus;

// Struct for a sequence of robot_common_interfaces__msg__RobotStatus.
typedef struct robot_common_interfaces__msg__RobotStatus__Sequence
{
  robot_common_interfaces__msg__RobotStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_common_interfaces__msg__RobotStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_
