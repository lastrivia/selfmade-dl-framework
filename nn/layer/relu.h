#pragma once

#include "base_layer.h"

class relu_layer : public nn_layer {
public:
    explicit relu_layer(bool in_place = false) : input_(1), in_place_(in_place) {}

    ~relu_layer() override = default;

    tensor forward_propagation(tensor &input) override {
        input_ = input;
        if (in_place_) {
            input.relu();
            return input;
        }
        else
            return relu(input);
    }

    tensor back_propagation(tensor &output_grad) override {
        return relu_mask(output_grad, input_);
    }

    std::vector<param> enum_params() override {
        return {};
    }

private:
    bool in_place_;
    tensor input_;
};
