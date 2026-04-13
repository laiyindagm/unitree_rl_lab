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

} // namespace isaaclab

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    const auto& joint_ids_map = env->robot->data.joint_ids_map;

    // Cache the list of sim-order joint indices that the policy controls.
    // For 15-DOF: deploy.yaml sets actions.JointPositionAction.joint_ids to
    // the 15 leg+waist joint indices within the full 29-joint sim array.
    // For 29-DOF (joint_ids: null): falls back to all joints.
    static std::vector<int> ctrl_ids;
    if (ctrl_ids.empty())
    {
        auto jids_node = env->cfg["actions"]["JointPositionAction"]["joint_ids"];
        if (jids_node && !jids_node.IsNull())
        {
            ctrl_ids = jids_node.as<std::vector<int>>();
        }
        else
        {
            for (int i = 0; i < static_cast<int>(joint_ids_map.size()); ++i)
                ctrl_ids.push_back(i);
        }
    }

    // processed_actions() returns position targets for the policy-controlled
    // joints only (15 values for the 15-DOF policy).  Arm motors are not
    // touched here; they keep the last position commanded by FixStand with
    // their PD gains, effectively holding the arms at the default pose.
    auto action = env->action_manager->processed_actions();
    for (int i = 0; i < static_cast<int>(ctrl_ids.size()); ++i)
    {
        int sdk_id = joint_ids_map[ctrl_ids[i]];
        lowcmd->msg_.motor_cmd()[sdk_id].q() = action[i];
    }
}
