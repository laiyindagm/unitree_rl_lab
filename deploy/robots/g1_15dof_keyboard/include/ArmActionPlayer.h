#pragma once
/**
 * ArmActionPlayer.h — YAML 驱动的上肢关键帧动作播放器 (header-only)
 *
 * 用途: 在 15-DOF 策略控制腿部的同时, 通过键盘触发上肢动作.
 *       策略只写 legs+waist, 本模块只写 arms, 互不干扰.
 *
 * G1 关节映射速查 (SDK motor index / sim index):
 *   SDK 0-5   (sim 0,3,6,9,13,17)        — Left Leg
 *   SDK 6-11  (sim 1,4,7,10,14,18)       — Right Leg
 *   SDK 12-14 (sim 2,5,8)                — Waist
 *   SDK 15-21 (sim 11,15,19,21,23,25,27) — Left Arm
 *       15=L_ShoulderPitch  16=L_ShoulderRoll  17=L_ShoulderYaw
 *       18=L_Elbow          19=L_WristRoll
 *       20=L_WristPitch     21=L_WristYaw
 *   SDK 22-28 (sim 12,16,20,22,24,26,28) — Right Arm
 *       22=R_ShoulderPitch  23=R_ShoulderRoll  24=R_ShoulderYaw
 *       25=R_Elbow          26=R_WristRoll
 *       27=R_WristPitch     28=R_WristYaw
 *
 * YAML 格式:
 *   ArmActions:
 *     wave_hello:
 *       key: "1"               # 触发按键
 *       kp: 40.0               # P 增益
 *       kd: 1.5                # D 增益
 *       motor_ids: [22,...,28]  # 受控 SDK motor index
 *       keyframes:
 *         - { t: 0.0, q: [values matching motor_ids count] }
 *         - { t: 1.5, q: [...] }
 */

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

class ArmActionPlayer
{
public:
    // ── 从 YAML 加载所有动作 ────────────────────────────────────────
    void load(const YAML::Node& node)
    {
        if (!node || !node.IsMap())
            return;

        for (auto it = node.begin(); it != node.end(); ++it) {
            Action act;
            act.name = it->first.as<std::string>();
            const auto& cfg = it->second;

            act.key       = cfg["key"].as<std::string>();
            act.kp        = cfg["kp"].as<float>(40.0f);
            act.kd        = cfg["kd"].as<float>(1.5f);
            act.motor_ids = cfg["motor_ids"].as<std::vector<int>>();

            const int n_joints = static_cast<int>(act.motor_ids.size());

            for (const auto& kf_node : cfg["keyframes"]) {
                Keyframe kf;
                kf.time = kf_node["t"].as<float>();
                kf.q    = kf_node["q"].as<std::vector<float>>();

                if (static_cast<int>(kf.q.size()) != n_joints) {
                    spdlog::error("[ArmAction] '{}' t={:.2f}: got {} values, need {} (motor_ids size)",
                                  act.name, kf.time, kf.q.size(), n_joints);
                    continue;
                }
                act.keyframes.push_back(std::move(kf));
            }

            std::sort(act.keyframes.begin(), act.keyframes.end(),
                      [](const Keyframe& a, const Keyframe& b) { return a.time < b.time; });

            if (act.keyframes.size() < 2) {
                spdlog::warn("[ArmAction] '{}' has <2 valid keyframes, skipped.", act.name);
                continue;
            }

            key_map_[act.key] = static_cast<int>(actions_.size());
            spdlog::info("[ArmAction] Loaded '{}' -> key='{}' ({} joints, {} keyframes, {:.1f}s)",
                         act.name, act.key, n_joints,
                         act.keyframes.size(), act.keyframes.back().time);
            actions_.push_back(std::move(act));
        }
    }

    // ── 按键触发 ────────────────────────────────────────────────────
    bool trigger(const std::string& key)
    {
        auto it = key_map_.find(key);
        if (it == key_map_.end())
            return false;

        current_ = it->second;
        elapsed_ = 0.0f;
        active_  = true;
        spdlog::info("[ArmAction] Playing '{}'", actions_[current_].name);
        return true;
    }

    void stop()
    {
        if (active_) {
            spdlog::info("[ArmAction] Stopped '{}'", actions_[current_].name);
            active_ = false;
        }
    }

    bool is_active() const { return active_; }
    bool has_actions() const { return !actions_.empty(); }

    // ── 推进时间 (每个控制周期调用) ─────────────────────────────────
    void update(float dt)
    {
        if (!active_)
            return;
        elapsed_ += dt;
        const auto& act = actions_[current_];
        if (act.keyframes.empty() || elapsed_ >= act.keyframes.back().time) {
            active_ = false;
            spdlog::info("[ArmAction] '{}' done.", act.name);
        }
    }

    /**
     * 将当前插值位置写入 lowcmd motor_cmd 数组.
     * MotorCmdArray 需支持 operator[](int), 返回带 q()/dq()/tau()/kp()/kd() 的对象.
     */
    template <typename MotorCmdArray>
    void apply(MotorCmdArray& motor_cmd) const
    {
        if (!active_)
            return;
        const auto& act = actions_[current_];
        auto q = interpolate(act.keyframes, elapsed_);
        for (int i = 0; i < static_cast<int>(act.motor_ids.size()); ++i) {
            int mid = act.motor_ids[i];
            motor_cmd[mid].q()   = q[i];
            motor_cmd[mid].dq()  = 0.0f;
            motor_cmd[mid].tau() = 0.0f;
            motor_cmd[mid].kp()  = act.kp;
            motor_cmd[mid].kd()  = act.kd;
        }
    }

    // ── 返回帮助字符串 (用于打印按键提示) ───────────────────────────
    std::string help_string() const
    {
        std::string s;
        for (const auto& a : actions_)
            s += "  " + a.key + ": " + a.name + "\n";
        return s;
    }

private:
    struct Keyframe
    {
        float time;
        std::vector<float> q;
    };

    struct Action
    {
        std::string name;
        std::string key;
        float kp = 40.0f;
        float kd = 1.5f;
        std::vector<int> motor_ids; // SDK motor indices
        std::vector<Keyframe> keyframes;
    };

    /// 余弦缓入缓出
    static float cosine_ease(float ratio)
    {
        ratio = std::clamp(ratio, 0.0f, 1.0f);
        return 0.5f * (1.0f - std::cos(static_cast<float>(M_PI) * ratio));
    }

    /// 在排好序的关键帧中做分段余弦插值
    static std::vector<float> interpolate(const std::vector<Keyframe>& kfs, float t)
    {
        if (kfs.empty())
            return {};
        if (t <= kfs.front().time)
            return kfs.front().q;
        if (t >= kfs.back().time)
            return kfs.back().q;

        for (size_t i = 0; i + 1 < kfs.size(); ++i) {
            if (t >= kfs[i].time && t < kfs[i + 1].time) {
                float seg_dt = kfs[i + 1].time - kfs[i].time;
                float alpha  = (seg_dt > 0.0f) ? cosine_ease((t - kfs[i].time) / seg_dt) : 1.0f;
                const int n  = static_cast<int>(kfs[i].q.size());
                std::vector<float> q(n);
                for (int j = 0; j < n; ++j)
                    q[j] = kfs[i].q[j] + alpha * (kfs[i + 1].q[j] - kfs[i].q[j]);
                return q;
            }
        }
        return kfs.back().q;
    }

    std::vector<Action> actions_;
    std::map<std::string, int> key_map_;
    int current_  = -1;
    float elapsed_ = 0.0f;
    bool active_   = false;
};
