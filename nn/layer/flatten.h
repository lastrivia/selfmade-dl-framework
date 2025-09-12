#pragma once

#include "base_layer.h"

class flatten_layer : public nn_layer {
public:
    flatten_layer() = default;

    ~flatten_layer() override = default;

    tensor forward_propagation(tensor &input) override {
        input_layout_ = input.layout();
        input.width_ = input.channels_ * input.height_ * input.width_;
        input.height_ = input.samples_;
        input.channels_ = 1;
        input.samples_ = 1;
        return input;
    }

    tensor back_propagation(tensor &output_grad) override {
        output_grad.samples_ = input_layout_.samples;
        output_grad.channels_ = input_layout_.channels;
        output_grad.height_ = input_layout_.height;
        output_grad.width_ = input_layout_.width;
        return output_grad;
    }

    std::vector<param> enum_params() override {
        return {};
    }

private:
    tensor::layout_t input_layout_{};
};
