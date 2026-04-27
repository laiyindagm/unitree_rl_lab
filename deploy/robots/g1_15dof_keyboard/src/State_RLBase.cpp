#include "FSM/State_RLBase.h"
#include "KeyboardTeleop.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <cmath>

namespace isaaclab
{

REGISTER_OBSERVATION(velocity_commands)
{
    const auto cmd = g_keyboard_teleop.command();
    return std::vector<float>{cmd[0], cmd[1], cmd[2]};
}

// V20a: 5-mode one-hot gait token (standing/pure_vx/pure_vy/pure_wz/joint).
// MUST mirror mdp.gait_mode_token in Python training code so the policy sees
// identical observations in sim and on hardware. Reads keyboard command,
// applies per-axis epsilon thresholds, returns one-hot vector.
REGISTER_OBSERVATION(gait_mode_token)
{
    const auto cmd = g_keyboard_teleop.command();
    const float eps_x = params["eps_x"] ? params["eps_x"].as<float>() : 0.1f;
    const float eps_y = params["eps_y"] ? params["eps_y"].as<float>() : 0.1f;
    const float eps_w = params["eps_w"] ? params["eps_w"].as<float>() : 0.1f;

    const bool vx_zero = std::abs(cmd[0]) < eps_x;
    const bool vy_zero = std::abs(cmd[1]) < eps_y;
    const bool wz_zero = std::abs(cmd[2]) < eps_w;

    const bool standing = vx_zero && vy_zero && wz_zero;
    const bool pure_vx  = (!vx_zero) && vy_zero && wz_zero;
    const bool pure_vy  = vx_zero && (!vy_zero) && wz_zero;
    const bool pure_wz  = vx_zero && vy_zero && (!wz_zero);
    const bool joint    = !(standing || pure_vx || pure_vy || pure_wz);

    return std::vector<float>{
        standing ? 1.0f : 0.0f,
        pure_vx  ? 1.0f : 0.0f,
        pure_vy  ? 1.0f : 0.0f,
        pure_wz  ? 1.0f : 0.0f,
        joint    ? 1.0f : 0.0f,
    };
}

// V20j: 3-mode one-hot gait token {standing, pure_wz, other}.
// MUST mirror mdp.gait_mode_token_3 in Python training code. Collapses
// pure_vx/pure_vy/joint into a single "other" bucket so joint envs'
// cmd_wz!=0 yaw signal can transfer (via shared subpolicy params) to
// pure_vx/pure_vy samples. Keeps {standing, pure_wz} isolated for
// dedicated subpolicies (qualitatively distinct objectives).
//
// Coexistence with the 5-mode gait_mode_token registration above is
// safe: deploy.yaml selects an observation by name, so any given
// policy.onnx uses exactly one of {gait_mode_token, gait_mode_token_3}
// (or neither, for V19f/V20i which omit the token entirely).
REGISTER_OBSERVATION(gait_mode_token_3)
{
    const auto cmd = g_keyboard_teleop.command();
    const float eps_x = params["eps_x"] ? params["eps_x"].as<float>() : 0.1f;
    const float eps_y = params["eps_y"] ? params["eps_y"].as<float>() : 0.1f;
    const float eps_w = params["eps_w"] ? params["eps_w"].as<float>() : 0.1f;

    const bool vx_zero = std::abs(cmd[0]) < eps_x;
    const bool vy_zero = std::abs(cmd[1]) < eps_y;
    const bool wz_zero = std::abs(cmd[2]) < eps_w;

    const bool standing = vx_zero && vy_zero && wz_zero;
    const bool pure_wz  = vx_zero && vy_zero && (!wz_zero);
    const bool other    = !(standing || pure_wz);

    return std::vector<float>{
        standing ? 1.0f : 0.0f,
        pure_wz  ? 1.0f : 0.0f,
        other    ? 1.0f : 0.0f,
    };
}


// Override gait_phase to support speed-adaptive mode (V9a).
// Detects walk_period in params -> speed-adaptive; otherwise falls back to
// fixed-period mode (original behaviour).
REGISTER_OBSERVATION(gait_phase)
{
    // Speed-adaptive mode: walk_period present in params
    if (params["walk_period"] && !params["walk_period"].IsNull()) {
        const float walk_period  = params["walk_period"].as<float>();
        const float run_period   = params["run_period"].as<float>(0.7f);
        const float speed_thresh = params["speed_threshold"].as<float>(0.8f);
        const float decay_factor = params["decay_factor"].as<float>(0.95f);
        const float still_thresh = params["standstill_threshold"].as<float>(0.1f);

        // Get commanded velocity from keyboard
        const auto cmd = g_keyboard_teleop.command();
        const float speed = std::sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1]);

        if (speed < still_thresh) {
            // Standstill: decay phase toward 0 -> "stop stepping" signal
            env->global_phase *= decay_factor;
        } else {
            // Speed-dependent period: slow walk -> walk_period, fast -> run_period
            const float alpha = std::min(speed / speed_thresh, 1.0f);
            const float period = walk_period - (walk_period - run_period) * alpha;
            env->global_phase += env->step_dt / period;
        }
        env->global_phase = std::fmod(env->global_phase, 1.0f);
    } else {
        // Fixed-period mode (backward compatible)
        const float period = params["period"].as<float>();
        env->global_phase += env->step_dt / period;
        env->global_phase = std::fmod(env->global_phase, 1.0f);
    }

    const float two_pi = 2.0f * static_cast<float>(M_PI);
    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * two_pi);
    obs[1] = std::cos(env->global_phase * two_pi);
    return obs;
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
