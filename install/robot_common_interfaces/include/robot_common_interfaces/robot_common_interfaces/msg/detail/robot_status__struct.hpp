// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from robot_common_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice

#ifndef ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_HPP_
#define ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__robot_common_interfaces__msg__RobotStatus __attribute__((deprecated))
#else
# define DEPRECATED__robot_common_interfaces__msg__RobotStatus __declspec(deprecated)
#endif

namespace robot_common_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotStatus_
{
  using Type = RobotStatus_<ContainerAllocator>;

  explicit RobotStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->battery_soc = 0.0f;
      this->battery_voltage = 0.0f;
      this->estop_active = false;
      this->motion_mode = "";
      this->hw_ok = false;
    }
  }

  explicit RobotStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    motion_mode(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->battery_soc = 0.0f;
      this->battery_voltage = 0.0f;
      this->estop_active = false;
      this->motion_mode = "";
      this->hw_ok = false;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _battery_soc_type =
    float;
  _battery_soc_type battery_soc;
  using _battery_voltage_type =
    float;
  _battery_voltage_type battery_voltage;
  using _estop_active_type =
    bool;
  _estop_active_type estop_active;
  using _motion_mode_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _motion_mode_type motion_mode;
  using _hw_ok_type =
    bool;
  _hw_ok_type hw_ok;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__battery_soc(
    const float & _arg)
  {
    this->battery_soc = _arg;
    return *this;
  }
  Type & set__battery_voltage(
    const float & _arg)
  {
    this->battery_voltage = _arg;
    return *this;
  }
  Type & set__estop_active(
    const bool & _arg)
  {
    this->estop_active = _arg;
    return *this;
  }
  Type & set__motion_mode(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->motion_mode = _arg;
    return *this;
  }
  Type & set__hw_ok(
    const bool & _arg)
  {
    this->hw_ok = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    robot_common_interfaces::msg::RobotStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const robot_common_interfaces::msg::RobotStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<robot_common_interfaces::msg::RobotStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<robot_common_interfaces::msg::RobotStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      robot_common_interfaces::msg::RobotStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<robot_common_interfaces::msg::RobotStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      robot_common_interfaces::msg::RobotStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<robot_common_interfaces::msg::RobotStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<robot_common_interfaces::msg::RobotStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<robot_common_interfaces::msg::RobotStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__robot_common_interfaces__msg__RobotStatus
    std::shared_ptr<robot_common_interfaces::msg::RobotStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__robot_common_interfaces__msg__RobotStatus
    std::shared_ptr<robot_common_interfaces::msg::RobotStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotStatus_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->battery_soc != other.battery_soc) {
      return false;
    }
    if (this->battery_voltage != other.battery_voltage) {
      return false;
    }
    if (this->estop_active != other.estop_active) {
      return false;
    }
    if (this->motion_mode != other.motion_mode) {
      return false;
    }
    if (this->hw_ok != other.hw_ok) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotStatus_

// alias to use template instance with default allocator
using RobotStatus =
  robot_common_interfaces::msg::RobotStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace robot_common_interfaces

#endif  // ROBOT_COMMON_INTERFACES__MSG__DETAIL__ROBOT_STATUS__STRUCT_HPP_
