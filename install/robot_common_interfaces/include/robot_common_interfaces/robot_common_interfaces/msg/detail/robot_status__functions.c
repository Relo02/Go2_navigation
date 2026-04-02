// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from robot_common_interfaces:msg/RobotStatus.idl
// generated code does not contain a copyright notice
#include "robot_common_interfaces/msg/detail/robot_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `motion_mode`
#include "rosidl_runtime_c/string_functions.h"

bool
robot_common_interfaces__msg__RobotStatus__init(robot_common_interfaces__msg__RobotStatus * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    robot_common_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // battery_soc
  // battery_voltage
  // estop_active
  // motion_mode
  if (!rosidl_runtime_c__String__init(&msg->motion_mode)) {
    robot_common_interfaces__msg__RobotStatus__fini(msg);
    return false;
  }
  // hw_ok
  return true;
}

void
robot_common_interfaces__msg__RobotStatus__fini(robot_common_interfaces__msg__RobotStatus * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // battery_soc
  // battery_voltage
  // estop_active
  // motion_mode
  rosidl_runtime_c__String__fini(&msg->motion_mode);
  // hw_ok
}

bool
robot_common_interfaces__msg__RobotStatus__are_equal(const robot_common_interfaces__msg__RobotStatus * lhs, const robot_common_interfaces__msg__RobotStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // battery_soc
  if (lhs->battery_soc != rhs->battery_soc) {
    return false;
  }
  // battery_voltage
  if (lhs->battery_voltage != rhs->battery_voltage) {
    return false;
  }
  // estop_active
  if (lhs->estop_active != rhs->estop_active) {
    return false;
  }
  // motion_mode
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->motion_mode), &(rhs->motion_mode)))
  {
    return false;
  }
  // hw_ok
  if (lhs->hw_ok != rhs->hw_ok) {
    return false;
  }
  return true;
}

bool
robot_common_interfaces__msg__RobotStatus__copy(
  const robot_common_interfaces__msg__RobotStatus * input,
  robot_common_interfaces__msg__RobotStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // battery_soc
  output->battery_soc = input->battery_soc;
  // battery_voltage
  output->battery_voltage = input->battery_voltage;
  // estop_active
  output->estop_active = input->estop_active;
  // motion_mode
  if (!rosidl_runtime_c__String__copy(
      &(input->motion_mode), &(output->motion_mode)))
  {
    return false;
  }
  // hw_ok
  output->hw_ok = input->hw_ok;
  return true;
}

robot_common_interfaces__msg__RobotStatus *
robot_common_interfaces__msg__RobotStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_common_interfaces__msg__RobotStatus * msg = (robot_common_interfaces__msg__RobotStatus *)allocator.allocate(sizeof(robot_common_interfaces__msg__RobotStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(robot_common_interfaces__msg__RobotStatus));
  bool success = robot_common_interfaces__msg__RobotStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
robot_common_interfaces__msg__RobotStatus__destroy(robot_common_interfaces__msg__RobotStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    robot_common_interfaces__msg__RobotStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
robot_common_interfaces__msg__RobotStatus__Sequence__init(robot_common_interfaces__msg__RobotStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_common_interfaces__msg__RobotStatus * data = NULL;

  if (size) {
    data = (robot_common_interfaces__msg__RobotStatus *)allocator.zero_allocate(size, sizeof(robot_common_interfaces__msg__RobotStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = robot_common_interfaces__msg__RobotStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        robot_common_interfaces__msg__RobotStatus__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
robot_common_interfaces__msg__RobotStatus__Sequence__fini(robot_common_interfaces__msg__RobotStatus__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      robot_common_interfaces__msg__RobotStatus__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

robot_common_interfaces__msg__RobotStatus__Sequence *
robot_common_interfaces__msg__RobotStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  robot_common_interfaces__msg__RobotStatus__Sequence * array = (robot_common_interfaces__msg__RobotStatus__Sequence *)allocator.allocate(sizeof(robot_common_interfaces__msg__RobotStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = robot_common_interfaces__msg__RobotStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
robot_common_interfaces__msg__RobotStatus__Sequence__destroy(robot_common_interfaces__msg__RobotStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    robot_common_interfaces__msg__RobotStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
robot_common_interfaces__msg__RobotStatus__Sequence__are_equal(const robot_common_interfaces__msg__RobotStatus__Sequence * lhs, const robot_common_interfaces__msg__RobotStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!robot_common_interfaces__msg__RobotStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
robot_common_interfaces__msg__RobotStatus__Sequence__copy(
  const robot_common_interfaces__msg__RobotStatus__Sequence * input,
  robot_common_interfaces__msg__RobotStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(robot_common_interfaces__msg__RobotStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    robot_common_interfaces__msg__RobotStatus * data =
      (robot_common_interfaces__msg__RobotStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!robot_common_interfaces__msg__RobotStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          robot_common_interfaces__msg__RobotStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!robot_common_interfaces__msg__RobotStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
