// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from robot_common_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

#ifndef ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__TRAITS_HPP_
#define ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "robot_common_interfaces/msg/detail/robot_status__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace robot_common_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotStatus & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: battery_soc
  {
    out << "battery_soc: ";
    rosidl_generator_traits::value_to_yaml(msg.battery_soc, out);
    out << ", ";
  }

  // member: battery_voltage
  {
    out << "battery_voltage: ";
    rosidl_generator_traits::value_to_yaml(msg.battery_voltage, out);
    out << ", ";
  }

  // member: estop_active
  {
    out << "estop_active: ";
    rosidl_generator_traits::value_to_yaml(msg.estop_active, out);
    out << ", ";
  }

  // member: motion_mode
  {
    out << "motion_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.motion_mode, out);
    out << ", ";
  }

  // member: hw_ok
  {
    out << "hw_ok: ";
    rosidl_generator_traits::value_to_yaml(msg.hw_ok, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: battery_soc
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "battery_soc: ";
    rosidl_generator_traits::value_to_yaml(msg.battery_soc, out);
    out << "\n";
  }

  // member: battery_voltage
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "battery_voltage: ";
    rosidl_generator_traits::value_to_yaml(msg.battery_voltage, out);
    out << "\n";
  }

  // member: estop_active
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "estop_active: ";
    rosidl_generator_traits::value_to_yaml(msg.estop_active, out);
    out << "\n";
  }

  // member: motion_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "motion_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.motion_mode, out);
    out << "\n";
  }

  // member: hw_ok
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "hw_ok: ";
    rosidl_generator_traits::value_to_yaml(msg.hw_ok, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotStatus & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace robot_common_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_common_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_common_interfaces::msg::RobotStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_common_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_common_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const robot_common_interfaces::msg::RobotStatus & msg)
{
  return robot_common_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<robot_common_interfaces::msg::RobotStatus>()
{
  return "robot_common_interfaces::msg::RobotStatus";
}

template<>
inline const char * name<robot_common_interfaces::msg::RobotStatus>()
{
  return "robot_common_interfaces/msg/RobotStatus";
}

template<>
struct has_fixed_size<robot_common_interfaces::msg::RobotStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<robot_common_interfaces::msg::RobotStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<robot_common_interfaces::msg::RobotStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__TRAITS_HPP_
