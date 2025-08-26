#pragma once

#include "base_layer.h"

class relu_layer : public nn_layer {
public:
    explicit relu_layer() : input_(1) {}

    ~relu_layer() override = default;

    tensor forward_propagation(const tensor &input) override {
        input_ = input;
        tensor result = input;
        for (int i = 0; i < result.rows(); ++i) {
            if (result(i) < 0.0)
                result(i) = 0.0;
        }
        return result;
    }

    tensor back_propagation(const tensor &output_grad) override {
        tensor input_grad = output_grad;
        for (int i = 0; i < input_grad.rows(); ++i) {
            if (input_(i) < 0.0)
                input_grad(i) = 0.0;
        }
        return input_grad;
    }

    std::vector<param *> enum_params() override {
        return {};
    }

private:
    tensor input_;
};
