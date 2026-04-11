#include <array>
#include <chrono>
#include <thread>
#include <sstream>

#include "FSM/State_FixStand.h"
#include "FSM/State_Passive.h"
#include "FSM/State_RLBase.h"
#include "KeyboardTeleop.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

namespace
{

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

std::string command_string()
{
    const auto cmd = g_keyboard_teleop.command();
    std::ostringstream oss;
    oss << "cmd = [vx=" << cmd[0] << ", vy=" << cmd[1] << ", wz=" << cmd[2] << "]";
    return oss.str();
}

} // namespace

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);
    FSMState::keyboard = std::shared_ptr<Keyboard>(new Keyboard(), [](Keyboard*) {});

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-15dof Keyboard Controller \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // G1 29dof hardware
    if (!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        return -1;
    }

    auto passive = std::make_shared<State_Passive>(1, "Passive");
    auto fix_stand = std::make_shared<State_FixStand>(2, "FixStand");
    auto velocity = std::make_shared<State_RLBase>(3, "Velocity");

    std::shared_ptr<BaseState> current_state = passive;
    auto transition_to = [&](const std::shared_ptr<BaseState>& next_state) {
        if (current_state == next_state) {
            return;
        }
        spdlog::info("State: {} -> {}", current_state->getStateString(), next_state->getStateString());
        current_state->exit();
        current_state = next_state;
        if (current_state->isState(passive->getState())) {
            g_keyboard_teleop.reset();
        }
        current_state->enter();
    };

    current_state->enter();
    const auto stand_duration = param::config["FSM"]["FixStand"]["ts"].as<std::vector<float>>().back();
    auto fixstand_enter_time = std::chrono::steady_clock::now();
    const auto teleop_cfg = param::config["Teleop"];
    const auto policy_dir = param::parser_policy_dir(param::config["FSM"]["Velocity"]["policy_dir"].as<std::string>());
    const auto ranges = YAML::LoadFile((policy_dir / "params" / "deploy.yaml").string())["commands"]["base_velocity"]["ranges"];
    const std::array<float, 2> lin_x_range {ranges["lin_vel_x"][0].as<float>(), ranges["lin_vel_x"][1].as<float>()};
    const std::array<float, 2> lin_y_range {ranges["lin_vel_y"][0].as<float>(), ranges["lin_vel_y"][1].as<float>()};
    const std::array<float, 2> yaw_range {ranges["ang_vel_z"][0].as<float>(), ranges["ang_vel_z"][1].as<float>()};
    const float step_x = teleop_cfg["lin_vel_x_step"].as<float>();
    const float step_y = teleop_cfg["lin_vel_y_step"].as<float>();
    const float step_yaw = teleop_cfg["ang_vel_z_step"].as<float>();

    std::cout << "Keys:\n";
    std::cout << "  f: Passive -> FixStand\n";
    std::cout << "  r: FixStand -> Velocity\n";
    std::cout << "  p: back to Passive\n";
    std::cout << "  w/s: increase/decrease forward velocity\n";
    std::cout << "  a/d: increase/decrease lateral velocity\n";
    std::cout << "  q/e: increase/decrease yaw velocity\n";
    std::cout << "  space: reset velocity command\n";
    std::cout << "  x: exit\n";

    using clock = std::chrono::steady_clock;
    const auto control_dt = std::chrono::microseconds(1000);
    auto wake_time = clock::now() + control_dt;

    while (true)
    {
        current_state->pre_run();

        if (FSMState::lowstate->isTimeout() && !current_state->isState(passive->getState())) {
            spdlog::warn("LowState timeout detected, returning to Passive.");
            transition_to(passive);
        }

        if (FSMState::keyboard->on_pressed)
        {
            const std::string key = FSMState::keyboard->key();
            if (key == "x") {
                break;
            }
            if (key == "p") {
                transition_to(passive);
            } else if (key == "f") {
                transition_to(fix_stand);
                fixstand_enter_time = clock::now();
            } else if (key == "r") {
                const auto stand_elapsed = std::chrono::duration<double>(clock::now() - fixstand_enter_time).count();
                if (!current_state->isState(fix_stand->getState())) {
                    spdlog::warn("Please enter FixStand first.");
                } else if (stand_elapsed < stand_duration) {
                    spdlog::warn("FixStand not finished yet. Wait {:.1f}s more.", stand_duration - stand_elapsed);
                } else {
                    transition_to(velocity);
                    spdlog::info("Keyboard velocity control enabled. {}", command_string());
                }
            } else if (key == " ") {
                g_keyboard_teleop.reset();
                spdlog::info("{}", command_string());
            } else if (current_state->isState(velocity->getState())) {
                if (key == "w") {
                    g_keyboard_teleop.nudge(step_x, 0.0f, 0.0f, lin_x_range, lin_y_range, yaw_range);
                } else if (key == "s") {
                    g_keyboard_teleop.nudge(-step_x, 0.0f, 0.0f, lin_x_range, lin_y_range, yaw_range);
                } else if (key == "a") {
                    g_keyboard_teleop.nudge(0.0f, step_y, 0.0f, lin_x_range, lin_y_range, yaw_range);
                } else if (key == "d") {
                    g_keyboard_teleop.nudge(0.0f, -step_y, 0.0f, lin_x_range, lin_y_range, yaw_range);
                } else if (key == "q") {
                    g_keyboard_teleop.nudge(0.0f, 0.0f, step_yaw, lin_x_range, lin_y_range, yaw_range);
                } else if (key == "e") {
                    g_keyboard_teleop.nudge(0.0f, 0.0f, -step_yaw, lin_x_range, lin_y_range, yaw_range);
                }

                if (key == "w" || key == "s" || key == "a" || key == "d" || key == "q" || key == "e") {
                    spdlog::info("{}", command_string());
                }
            }
        }

        current_state->run();
        current_state->post_run();

        std::this_thread::sleep_until(wake_time);
        wake_time += control_dt;
    }

    current_state->exit();
    g_keyboard_teleop.reset();
    return 0;
}
