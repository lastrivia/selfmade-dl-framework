#pragma once

#include <random>

#include "base_layer.h"

class conv_layer : public nn_layer {
public:
    conv_layer(
        const size_t channels_in, const size_t channels_out,
        const size_t kernel_height, const size_t kernel_width,
        const size_t height_padding, const size_t width_padding
    ) :
        kernel_(channels_out, channels_in, kernel_height, kernel_width),
        kernel_grad_(channels_out, channels_in, kernel_height, kernel_width),
        bias_(1, channels_out, 1, 1),
        bias_grad_(1, channels_out, 1, 1),
        height_padding_(height_padding), width_padding_(width_padding) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, sqrt(2.0 / static_cast<double>(channels_in * kernel_height * kernel_width)));
        for (size_t i = 0; i < channels_out * channels_in * kernel_height * kernel_width; ++i)
            kernel_.at(i) = static_cast<float>(dis(gen));
        broadcast(bias_, 0.0f);
    }

    ~conv_layer() override = default;

    tensor forward_propagation(tensor &input) override {
        input_ = input;
        return conv(input, kernel_, bias_, height_padding_, width_padding_);
    }

    tensor back_propagation(tensor &output_grad) override {
        kernel_grad_ = conv_kernel_grad(input_, output_grad, height_padding_, width_padding_);
        // bias_grad_ = sum_by_channel(output_grad);
        bias_grad_ = sum(output_grad, {0, 1, 3});
        return conv_input_grad(output_grad, kernel_, height_padding_, width_padding_);
    }

    std::vector<param> enum_params() override {
        return {{kernel_, kernel_grad_}, {bias_, bias_grad_}};
    }

private:
    tensor kernel_, kernel_grad_;
    tensor bias_, bias_grad_;
    tensor input_;

    size_t height_padding_, width_padding_;
};
