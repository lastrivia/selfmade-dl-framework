#pragma once

#include <random>

#include "tensor.h"
#include "except.h"
#include "base_classical.h"

class GaussianNB : public BaseClassicalAlgorithm {
public:
    GaussianNB(DeviceDesc device) : device_(device) {}

    void fit(Tensor features, Tensor labels) override {
        no_grad_lock lock;

        size_t n_samples = features->shape().lengths[1];
        if (n_samples != labels->shape().lengths[1])
            throw FatalExcept("sample counts in features and labels mismatch", __FILE__, __LINE__);
        size_t n_features = features->shape().lengths[0];
        size_t n_classes = labels->shape().lengths[0];

        Tensor n_per_class = sum(labels, {1}) + 1e-8f; // [1, n_classes]

        Tensor mean = div_broadcast(
            matmul<true, false>(features, labels),
            n_per_class
        ); // [n_features, n_classes]

        Tensor mean_s = matmul<false, true>(labels, mean); // [n_samples, n_features]

        Tensor var = div_broadcast(
                         matmul<true, false>(
                             square(features - mean_s),
                             labels
                         ),
                         n_per_class
                     ) + 1e-8f; // [n_features, n_classes]

        Tensor loglik_priori_ = log(n_per_class) - logf(static_cast<float>(n_samples)); // [1, n_classes]

        Tensor log_2pi_var_ = sum(log(var * (2.0f * 3.1415926535897932f)), {1}); // [1, n_classes]

        A_ = pow(var, -1.0f); // [n_features, n_classes] 1 / var
        B_ = mean * A_; // [n_features, n_classes] mean / var
        Tensor C = mean * B_; // [n_features, n_classes] mean^2 / var

        S_ = loglik_priori_ + (log_2pi_var_ + sum(C, {1})) * -0.5f; // [1, n_classes]
    }

    Tensor predict(const Tensor &features) override {
        no_grad_lock lock;

        Tensor loglik_posterior = add_broadcast(
            matmul(square(features), A_) * -0.5f + matmul(features, B_),
            S_
        );

        return loglik_posterior;
    }

private:
    Tensor A_, B_, S_;
    DeviceDesc device_;
};
