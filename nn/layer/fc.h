#pragma once

#include "base_layer.h"

class fc_layer : public nn_layer {
public:
    fc_layer(const int input_size, const int output_size): weight_(output_size, input_size),
                                                           bias_(output_size),
                                                           input_(input_size) {
        random_init_he();
    }

    ~fc_layer() override = default;

    tensor forward_propagation(const tensor &input) override {
        input_ = input;
        return weight_.data * input + bias_.data;
    }

    void random_init_he() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, sqrt(2.0 / static_cast<double>(input_.rows())));
        for (int i = 0; i < weight_.rows(); ++i)
            for (int j = 0; j < weight_.cols(); ++j)
                weight_.data(i, j) = static_cast<float>(dis(gen));
        for (int i = 0; i < bias_.rows(); ++i)
            bias_.data(i) = 0.0f;
    }

    tensor back_propagation(const tensor &output_grad) override {
        weight_.grad = output_grad * ~input_;
        bias_.grad = output_grad;
        const tensor input_grad = ~(weight_.data) * output_grad;
        return input_grad;
    }

    std::vector<param *> enum_params() override {
        return {&weight_, &bias_};
    }

    // debug
    void save_to(const std::string &path) const {
        std::ofstream file(path);
        for (int i = 0; i < weight_.rows(); ++i) {
            for (int j = 0; j < weight_.cols(); ++j) {
                file << weight_.data(i, j) << ", ";
            }
            file << ", " << bias_.data(i) << std::endl;
        }
    }

private:
    param weight_, bias_;
    tensor input_;
};
