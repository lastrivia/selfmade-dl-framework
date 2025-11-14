#pragma once

#include "base_optimizer.h"

class SgdOptimizer : public Optimizer {
public:
    explicit SgdOptimizer(float init_learning_rate) : Optimizer(init_learning_rate) {}

    ~SgdOptimizer() override = default;

    void register_layer(Layer &layer) override {
        std::vector<Tensor> layer_params = layer.enum_params();
        for (auto &p: layer_params) {
            params_.push_back(p);
        }
    }

    void step() override {
        for (auto &p: params_) {
            if (!p->grad_data_)
                continue;

            switch (p->dtype_) {
            case ScalarType::fp32: {
                const Backend &k = dispatch_kernel(p->device_);
                size_t size = p->shape_.size;
                Workspace tmp(size * sizeof(float), p->device_);

                k.mul_scalar_fp32(size, tmp, p->grad_data_, learning_rate_);
                k.sub_ewise_fp32(size, p->data_, p->data_, tmp);
                break;
            }
            case ScalarType::int32: {
                throw FatalExcept("tensor: adam does not support data type int32", __FILE__, __LINE__);
                break;
            }
            }
        }
    }

    void zero_grad() override {
        for (auto &p: params_) {
            if (!p->grad_data_)
                continue;
            p->zero_grad();
        }
    }

private:
    std::vector<Tensor> params_;
};
