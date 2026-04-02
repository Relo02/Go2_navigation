/**
 * sport_client.hpp
 * Go2 Sport Mode API client.
 *
 * Encodes sport-mode commands as unitree_api/Request messages.
 * The caller is responsible for publishing the request on /api/sport/request.
 *
 * Source adapted from go2_sport_api in /ws/autonomy_stack_go2.
 * Uses nlohmann/json (system package, MIT licence) for parameter serialisation.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "unitree_api/msg/request.hpp"

namespace go2_bringup
{

// API command IDs (from Unitree Go2 SDK documentation)
static constexpr int32_t SPORT_API_DAMP             = 1001;
static constexpr int32_t SPORT_API_BALANCESTAND      = 1002;
static constexpr int32_t SPORT_API_STOPMOVE          = 1003;
static constexpr int32_t SPORT_API_STANDUP           = 1004;
static constexpr int32_t SPORT_API_STANDDOWN         = 1005;
static constexpr int32_t SPORT_API_RECOVERYSTAND     = 1006;
static constexpr int32_t SPORT_API_EULER             = 1007;
static constexpr int32_t SPORT_API_MOVE              = 1008;
static constexpr int32_t SPORT_API_SIT               = 1009;
static constexpr int32_t SPORT_API_RISEDSIT          = 1010;
static constexpr int32_t SPORT_API_SWITCHGAIT        = 1011;
static constexpr int32_t SPORT_API_TRIGGER           = 1012;
static constexpr int32_t SPORT_API_BODYHEIGHT        = 1013;
static constexpr int32_t SPORT_API_FOOTRAISEHEIGHT   = 1014;
static constexpr int32_t SPORT_API_SPEEDLEVEL        = 1015;
static constexpr int32_t SPORT_API_HELLO             = 1016;
static constexpr int32_t SPORT_API_STRETCH           = 1017;
static constexpr int32_t SPORT_API_TRAJECTORYFOLLOW  = 1018;
static constexpr int32_t SPORT_API_CONTINUOUSGAIT    = 1019;
static constexpr int32_t SPORT_API_SWITCHJOYSTICK    = 1027;
static constexpr int32_t SPORT_API_CLASSICWALK       = 2049;
static constexpr int32_t SPORT_API_ECONOMICGAIT      = 1063;

struct PathPoint
{
  float time_from_start;
  float x, y, yaw;
  float vx, vy, vyaw;
};

/**
 * SportClient
 * Fills a unitree_api::msg::Request with the correct api_id and JSON parameter.
 * All methods write into the supplied `req` reference; no internal state.
 */
class SportClient
{
public:
  void Damp(unitree_api::msg::Request & req) const;
  void BalanceStand(unitree_api::msg::Request & req) const;
  void StopMove(unitree_api::msg::Request & req) const;
  void StandUp(unitree_api::msg::Request & req) const;
  void StandDown(unitree_api::msg::Request & req) const;
  void RecoveryStand(unitree_api::msg::Request & req) const;
  void Euler(unitree_api::msg::Request & req, float roll, float pitch, float yaw) const;

  /**
   * Move — primary navigation command.
   * @param vx    forward velocity [m/s]  (positive = forward)
   * @param vy    lateral velocity [m/s]  (positive = left)
   * @param vyaw  angular velocity [rad/s] (positive = counter-clockwise)
   */
  void Move(unitree_api::msg::Request & req, float vx, float vy, float vyaw) const;

  void Sit(unitree_api::msg::Request & req) const;
  void RiseSit(unitree_api::msg::Request & req) const;
  void SwitchGait(unitree_api::msg::Request & req, int gait) const;
  void Trigger(unitree_api::msg::Request & req) const;
  void BodyHeight(unitree_api::msg::Request & req, float height) const;
  void FootRaiseHeight(unitree_api::msg::Request & req, float height) const;
  void SpeedLevel(unitree_api::msg::Request & req, int level) const;
  void Hello(unitree_api::msg::Request & req) const;
  void Stretch(unitree_api::msg::Request & req) const;
  void TrajectoryFollow(
    unitree_api::msg::Request & req,
    const std::vector<PathPoint> & path) const;
  void SwitchJoystick(unitree_api::msg::Request & req, bool flag) const;
  void ContinuousGait(unitree_api::msg::Request & req, bool flag) const;
  void ClassicWalk(unitree_api::msg::Request & req, bool flag) const;
  void EconomicGait(unitree_api::msg::Request & req) const;

private:
  static void set_id(unitree_api::msg::Request & req, int32_t id);
};

}  // namespace go2_bringup
