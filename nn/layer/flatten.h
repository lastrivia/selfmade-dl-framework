#pragma once

#include "base_layer.h"

class FlattenLayer : public Layer {
public:
    FlattenLayer() = default;

    ~FlattenLayer() override = default;

    Tensor operator()(const Tensor &x) override {
        // temporary, todo use tensor_view to replace this
        return flatten(x);
    }

    std::vector<Tensor> enum_params() override {
        return {};
    }

    void to_device(DeviceDesc device) override {}
};
