// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
    // Velocity prediction output [vx, vy, wz] from velocity_head.
    // Non-empty only when the ONNX model exports a "velocity_pred" output
    // (i.e. TransformerLatentModel policies; empty for plain MLP policies).
    std::vector<float> velocity_pred;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Load ALL output names and shapes (supports 1 or 2 outputs)
        const size_t n_out = session->GetOutputCount();
        for (size_t i = 0; i < n_out; ++i) {
            Ort::TypeInfo ot = session->GetOutputTypeInfo(i);
            output_shapes.push_back(ot.GetTensorTypeAndShapeInfo().GetShape());
            auto oname = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(oname.release());
        }

        // First output: actions
        action.resize(output_shapes[0][1]);

        // Second output (optional): velocity_pred [vx, vy, wz]
        if (n_out >= 2) {
            velocity_pred.resize(output_shapes[1][1], 0.0f);
        }
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<static_cast<int>(input_names.size()); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model (request all registered outputs)
        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());

        std::lock_guard<std::mutex> lock(act_mtx_);

        // Copy action output
        auto floatarr = output_tensors[0].GetTensorMutableData<float>();
        std::memcpy(action.data(), floatarr, output_shapes[0][1] * sizeof(float));

        // Copy velocity_pred output if present
        if (output_tensors.size() >= 2) {
            auto vel_arr = output_tensors[1].GetTensorMutableData<float>();
            std::memcpy(velocity_pred.data(), vel_arr, output_shapes[1][1] * sizeof(float));
        }

        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<std::vector<int64_t>> output_shapes;
};
};
