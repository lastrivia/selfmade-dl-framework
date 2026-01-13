#pragma once

#include <random>

#include "tensor.h"
#include "except.h"
#include "base_classical.h"
#include "optimizer/sgd.h"

class LogisticRegression : public BaseClassicalAlgorithm {
public:
    LogisticRegression(float C, float learning_rate, size_t iter, size_t seed, DeviceDesc device) :
        C_(C), learning_rate_(learning_rate), iter_(iter), seed_(seed), device_(device) {}


    void fit(Tensor features, Tensor labels) override {
        size_t n_samples = features->shape().lengths[1];
        if (n_samples != labels->shape().lengths[1])
            throw FatalExcept("Sample counts in features and labels mismatch", __FILE__, __LINE__);
        size_t n_features = features->shape().lengths[0];
        size_t n_classes = labels->shape().lengths[0];

        weight_ = Tensor({n_features, n_classes});
        // std::random_device rd;
        std::mt19937 gen(seed_);
        std::normal_distribution<> dis(0.0, sqrt(2.0 / static_cast<double>(n_features + n_classes)));
        for (size_t i = 0; i < n_features * n_classes; ++i)
            weight_.at(i) = static_cast<float>(dis(gen));
        weight_->requires_grad(true);

        bias_ = Tensor({n_classes});
        bias_.fill(0.0f);
        bias_->requires_grad(true);

        weight_.to_device(device_);
        bias_.to_device(device_);

        {
            // training on device
            SgdOptimizer optimizer(learning_rate_);
            optimizer.register_tensor(weight_);
            optimizer.register_tensor(bias_);

            SgdOptimizer regularizer(learning_rate_ * 0.5 / C_);
            regularizer.register_tensor(weight_);

            for (size_t i = 0; i < iter_; ++i) {
                Tensor logits = add_broadcast(matmul(features, weight_), bias_);
                Tensor loss = cross_entropy(logits, labels);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                Tensor l2 = square(weight_);

                regularizer.zero_grad();
                l2.backward();
                regularizer.step();
            }
        }
    }

    Tensor predict(const Tensor &features) override {
        return add_broadcast(matmul(features, weight_), bias_);
    }

private:
    Tensor weight_, bias_;
    float C_;
    float learning_rate_;
    size_t iter_;
    size_t seed_;
    DeviceDesc device_;
};
