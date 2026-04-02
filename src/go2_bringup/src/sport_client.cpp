/**
 * sport_client.cpp
 * Implementation of SportClient.
 *
 * Uses nlohmann/json (system package, MIT licence) for parameter serialisation.
 * Install: sudo apt install nlohmann-json3-dev
 */

#include "go2_bringup/sport_client.hpp"
#include <nlohmann/json.hpp>

namespace go2_bringup
{

void SportClient::set_id(unitree_api::msg::Request & req, int32_t id)
{
  req.header.identity.api_id = id;
}

void SportClient::Damp(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_DAMP);
}

void SportClient::BalanceStand(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_BALANCESTAND);
}

void SportClient::StopMove(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_STOPMOVE);
}

void SportClient::StandUp(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_STANDUP);
}

void SportClient::StandDown(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_STANDDOWN);
}

void SportClient::RecoveryStand(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_RECOVERYSTAND);
}

void SportClient::Euler(
  unitree_api::msg::Request & req, float roll, float pitch, float yaw) const
{
  nlohmann::json js;
  js["x"] = roll;
  js["y"] = pitch;
  js["z"] = yaw;
  req.parameter = js.dump();
  set_id(req, SPORT_API_EULER);
}

void SportClient::Move(
  unitree_api::msg::Request & req, float vx, float vy, float vyaw) const
{
  nlohmann::json js;
  js["x"] = vx;
  js["y"] = vy;
  js["z"] = vyaw;
  req.parameter = js.dump();
  set_id(req, SPORT_API_MOVE);
}

void SportClient::Sit(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_SIT);
}

void SportClient::RiseSit(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_RISEDSIT);
}

void SportClient::SwitchGait(unitree_api::msg::Request & req, int gait) const
{
  nlohmann::json js;
  js["data"] = gait;
  req.parameter = js.dump();
  set_id(req, SPORT_API_SWITCHGAIT);
}

void SportClient::Trigger(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_TRIGGER);
}

void SportClient::BodyHeight(unitree_api::msg::Request & req, float height) const
{
  nlohmann::json js;
  js["data"] = height;
  req.parameter = js.dump();
  set_id(req, SPORT_API_BODYHEIGHT);
}

void SportClient::FootRaiseHeight(unitree_api::msg::Request & req, float height) const
{
  nlohmann::json js;
  js["data"] = height;
  req.parameter = js.dump();
  set_id(req, SPORT_API_FOOTRAISEHEIGHT);
}

void SportClient::SpeedLevel(unitree_api::msg::Request & req, int level) const
{
  nlohmann::json js;
  js["data"] = level;
  req.parameter = js.dump();
  set_id(req, SPORT_API_SPEEDLEVEL);
}

void SportClient::Hello(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_HELLO);
}

void SportClient::Stretch(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_STRETCH);
}

void SportClient::TrajectoryFollow(
  unitree_api::msg::Request & req,
  const std::vector<PathPoint> & path) const
{
  nlohmann::json js_path = nlohmann::json::array();
  const size_t n = std::min(path.size(), size_t{30});
  for (size_t i = 0; i < n; ++i) {
    nlohmann::json pt;
    pt["t_from_start"] = path[i].time_from_start;
    pt["x"]            = path[i].x;
    pt["y"]            = path[i].y;
    pt["yaw"]          = path[i].yaw;
    pt["vx"]           = path[i].vx;
    pt["vy"]           = path[i].vy;
    pt["vyaw"]         = path[i].vyaw;
    js_path.push_back(pt);
  }
  req.parameter = js_path.dump();
  set_id(req, SPORT_API_TRAJECTORYFOLLOW);
}

void SportClient::SwitchJoystick(unitree_api::msg::Request & req, bool flag) const
{
  nlohmann::json js;
  js["data"] = flag;
  req.parameter = js.dump();
  set_id(req, SPORT_API_SWITCHJOYSTICK);
}

void SportClient::ContinuousGait(unitree_api::msg::Request & req, bool flag) const
{
  nlohmann::json js;
  js["data"] = flag;
  req.parameter = js.dump();
  set_id(req, SPORT_API_CONTINUOUSGAIT);
}

void SportClient::ClassicWalk(unitree_api::msg::Request & req, bool flag) const
{
  nlohmann::json js;
  js["data"] = flag;
  req.parameter = js.dump();
  set_id(req, SPORT_API_CLASSICWALK);
}

void SportClient::EconomicGait(unitree_api::msg::Request & req) const
{
  set_id(req, SPORT_API_ECONOMICGAIT);
}

}  // namespace go2_bringup
