// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_common_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

#ifndef ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_
#define ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_common_interfaces/msg/detail/robot_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_common_interfaces
{

namespace msg
{

namespace builder
{

class Init_RobotStatus_hw_ok
{
public:
  explicit Init_RobotStatus_hw_ok(::robot_common_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  ::robot_common_interfaces::msg::RobotStatus hw_ok(::robot_common_interfaces::msg::RobotStatus::_hw_ok_type arg)
  {
    msg_.hw_ok = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_common_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_motion_mode
{
public:
  explicit Init_RobotStatus_motion_mode(::robot_common_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_hw_ok motion_mode(::robot_common_interfaces::msg::RobotStatus::_motion_mode_type arg)
  {
    msg_.motion_mode = std::move(arg);
    return Init_RobotStatus_hw_ok(msg_);
  }

private:
  ::robot_common_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_estop_active
{
public:
  explicit Init_RobotStatus_estop_active(::robot_common_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_motion_mode estop_active(::robot_common_interfaces::msg::RobotStatus::_estop_active_type arg)
  {
    msg_.estop_active = std::move(arg);
    return Init_RobotStatus_motion_mode(msg_);
  }

private:
  ::robot_common_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_battery_voltage
{
public:
  explicit Init_RobotStatus_battery_voltage(::robot_common_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_estop_active battery_voltage(::robot_common_interfaces::msg::RobotStatus::_battery_voltage_type arg)
  {
    msg_.battery_voltage = std::move(arg);
    return Init_RobotStatus_estop_active(msg_);
  }

private:
  ::robot_common_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_battery_soc
{
public:
  explicit Init_RobotStatus_battery_soc(::robot_common_interfaces::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_battery_voltage battery_soc(::robot_common_interfaces::msg::RobotStatus::_battery_soc_type arg)
  {
    msg_.battery_soc = std::move(arg);
    return Init_RobotStatus_battery_voltage(msg_);
  }

private:
  ::robot_common_interfaces::msg::RobotStatus msg_;
};

class Init_RobotStatus_header
{
public:
  Init_RobotStatus_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotStatus_battery_soc header(::robot_common_interfaces::msg::RobotStatus::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_RobotStatus_battery_soc(msg_);
  }

private:
  ::robot_common_interfaces::msg::RobotStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_common_interfaces::msg::RobotStatus>()
{
  return robot_common_interfaces::msg::builder::Init_RobotStatus_header();
}

}  // namespace robot_common_interfaces

#endif  // ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_
