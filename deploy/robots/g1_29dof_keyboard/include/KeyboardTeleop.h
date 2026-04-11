#pragma once

#include <array>
#include <algorithm>
#include <mutex>

class KeyboardTeleop
{
public:
    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        command_ = {0.0f, 0.0f, 0.0f};
    }

    void nudge(float dx, float dy, float dyaw, const std::array<float, 2>& x_range,
               const std::array<float, 2>& y_range, const std::array<float, 2>& yaw_range)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        command_[0] = std::clamp(command_[0] + dx, x_range[0], x_range[1]);
        command_[1] = std::clamp(command_[1] + dy, y_range[0], y_range[1]);
        command_[2] = std::clamp(command_[2] + dyaw, yaw_range[0], yaw_range[1]);
    }

    std::array<float, 3> command() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return command_;
    }

private:
    mutable std::mutex mutex_;
    std::array<float, 3> command_ {0.0f, 0.0f, 0.0f};
};

inline KeyboardTeleop g_keyboard_teleop;