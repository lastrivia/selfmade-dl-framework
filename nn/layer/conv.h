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
        kernel_({channels_out, channels_in, kernel_height, kernel_width}),
        bias_({1, channels_out, 1, 1}),
        height_padding_(height_padding), width_padding_(width_padding) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, sqrt(2.0 / static_cast<double>(channels_in * kernel_height * kernel_width)));
        for (size_t i = 0; i < channels_out * channels_in * kernel_height * kernel_width; ++i)
            kernel_.at(i) = static_cast<float>(dis(gen));
        bias_.fill(0.0f);

        kernel_->requires_grad(true);
        bias_->requires_grad(true);
    }

    ~conv_layer() override = default;

    tensor operator()(const tensor &x) override {
        return conv(x, kernel_, bias_, height_padding_, width_padding_);
    }

    std::vector<tensor> enum_params() override {
        return {kernel_, bias_};
    }

    void to_device(device_desc device) override {
        kernel_.to_device(device);
        bias_.to_device(device);
    }

private:
    tensor kernel_, bias_;
    size_t height_padding_, width_padding_;
};
