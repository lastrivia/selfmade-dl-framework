#pragma once

#include "base_optimizer.h"

class sgd_optimizer : public nn_optimizer {
public:
    explicit sgd_optimizer(float init_learning_rate) : nn_optimizer(init_learning_rate) {}

    ~sgd_optimizer() override = default;

    void register_layer(nn_layer &layer) override {
        std::vector<tensor> layer_params = layer.enum_params();
        for (auto &p: layer_params) {
            params_.push_back(p);
        }
    }

    void step() override {
        for (auto &p: params_) {
            if (!p->grad_data_)
                continue;

            switch (p->dtype_) {
            case data_type::fp32: {
                const kernel &k = dispatch_kernel(p->device_);
                size_t size = p->shape_.size;
                workspace tmp(size * sizeof(float), p->device_);

                k.mul_scalar_fp32(size, tmp, p->grad_data_, learning_rate_);
                k.sub_ewise_fp32(size, p->data_, p->data_, tmp);
                break;
            }
            case data_type::int32: {
                throw nn_except("tensor: adam does not support data type int32", __FILE__, __LINE__);
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
    std::vector<tensor> params_;
};
