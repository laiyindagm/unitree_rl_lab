#include "FSM/State_RLBase.h"
#include "KeyboardTeleop.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

namespace isaaclab
{

REGISTER_OBSERVATION(velocity_commands)
{
    const auto cmd = g_keyboard_teleop.command();
    return std::vector<float>{cmd[0], cmd[1], cmd[2]};
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // Build action-to-motor mapping
    const auto& joint_ids_map = env->robot->data.joint_ids_map;
    auto action_cfg = deploy_cfg["actions"];
    for (auto it = action_cfg.begin(); it != action_cfg.end(); ++it) {
        auto term_cfg = it->second;
        if (!term_cfg["joint_ids"].IsNull()) {
            auto joint_ids = term_cfg["joint_ids"].as<std::vector<int>>();
            for (int id : joint_ids) {
                action_motor_ids_.push_back(static_cast<int>(joint_ids_map[id]));
            }
        } else {
            for (int i = 0; i < static_cast<int>(joint_ids_map.size()); i++) {
                action_motor_ids_.push_back(static_cast<int>(joint_ids_map[i]));
            }
        }
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for (int i = 0; i < static_cast<int>(action_motor_ids_.size()); i++) {
        lowcmd->msg_.motor_cmd()[action_motor_ids_[i]].q() = action[i];
    }
}
