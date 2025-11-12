#pragma once

#include "base_layer.h"

class flatten_layer : public nn_layer {
public:
    flatten_layer() = default;

    ~flatten_layer() override = default;

    tensor operator()(const tensor &x) override {
        // temporary, todo use tensor_view to replace this
        return flatten(x);
    }

    std::vector<tensor> enum_params() override {
        return {};
    }

    void to_device(device_desc device) override {}
};
