#pragma once

#include "base_layer.h"

class relu_layer : public nn_layer {
public:
    explicit relu_layer() : input_(1) {}

    ~relu_layer() override = default;

    tensor forward_propagation(const tensor &input) override {
        input_ = input;
        return relu(input);
    }

    tensor back_propagation(const tensor &output_grad) override {
        return relu_mask(output_grad, input_);
    }

    std::vector<param> enum_params() override {
        return {};
    }

private:
    tensor input_;
};
