#pragma once

#include "base_layer.h"

class ReluLayer : public Layer {
public:
    ReluLayer() = default;

    ~ReluLayer() override = default;

    Tensor operator()(const Tensor &x) override {
        return relu(x);
    }

    std::vector<Tensor> enum_params() override {
        return {};
    }

    void to_device(DeviceDesc device) override {}
};
