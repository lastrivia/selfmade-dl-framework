#pragma once

#include <random>

#include "base_layer.h"

class FCLayer : public Layer {
public:
    FCLayer(const size_t input_size, const size_t output_size) :
        weight_({input_size, output_size}),
        bias_({output_size}) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, sqrt(2.0 / static_cast<double>(input_size)));
        for (size_t i = 0; i < input_size; ++i)
            for (size_t j = 0; j < output_size; ++j)
                weight_.at(i, j) = static_cast<float>(dis(gen));
        bias_.fill(0.0f);

        weight_->requires_grad(true);
        bias_->requires_grad(true);
    }

    ~FCLayer() override = default;

    Tensor operator()(const Tensor &x) override {
        return add_broadcast(matmul(x, weight_), bias_);
    }

    std::vector<Tensor> enum_params() override {
        return {weight_, bias_};
    }

    void to_device(DeviceDesc device) override {
        weight_.to_device(device);
        bias_.to_device(device);
    }

private:
    Tensor weight_, bias_;
};
