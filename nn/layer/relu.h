#pragma once

#include "base_layer.h"

class relu_layer : public nn_layer {
public:
    relu_layer() = default;

    ~relu_layer() override = default;

    tensor operator()(const tensor &x) override {
        return relu(x);
    }

    std::vector<tensor> enum_params() override {
        return {};
    }

    void to_device(device_desc device) override {}
};
