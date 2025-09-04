#pragma once

#include <random>

#include "base_layer.h"

class fc_layer : public nn_layer {
public:
    fc_layer(const int input_size, const int output_size) :
        weight_(input_size, output_size),
        weight_grad_(input_size, output_size),
        bias_(1, output_size),
        bias_grad_(1, output_size),
        input_(1) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, sqrt(2.0 / static_cast<double>(input_size)));
        for (int i = 0; i < input_size; ++i)
            for (int j = 0; j < output_size; ++j)
                weight_.at(i, j) = static_cast<float>(dis(gen));
        broadcast(bias_, 0.0f);
    }

    ~fc_layer() override = default;

    tensor forward_propagation(const tensor &input) override {
        input_ = input;
        return add_tile(matmul(input, weight_), bias_);
    }

    tensor back_propagation(const tensor &output_grad) override {
        weight_grad_ = matmul<true, false>(input_, output_grad);
        bias_grad_ = sum_rows(output_grad);
        return matmul<false, true>(output_grad, weight_);
    }

    std::vector<param> enum_params() override {
        return {{weight_, weight_grad_}, {bias_, bias_grad_}};
    }

private:
    tensor weight_, weight_grad_;
    tensor bias_, bias_grad_;
    tensor input_;
};
