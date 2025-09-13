#pragma once

#include "base_layer.h"

class maxpool_layer : public nn_layer {
public:
    maxpool_layer(const size_t h_stride, const size_t w_stride)
        : h_stride_(h_stride), w_stride_(w_stride), input_height_(), input_width_() {}

    ~maxpool_layer() override = default;

    tensor forward_propagation(tensor &input) override {
        mask_.bind_layout(input);
        input_height_ = input.height();
        input_width_ = input.width();
        return maxpool(input, mask_, h_stride_, w_stride_);
    }

    tensor back_propagation(tensor &output_grad) override {
        return maxpool_backward(output_grad, mask_, input_height_, input_width_, h_stride_, w_stride_);
    }

    std::vector<param> enum_params() override {
        return {};
    }

private:
    size_t h_stride_, w_stride_;
    size_t input_height_, input_width_;
    tensor_mask mask_;
};
