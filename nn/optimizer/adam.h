#pragma once

#include "base_optimizer.h"

class AdamOptimizer : public Optimizer {
    struct adam_param {
        Tensor target;
        Data first_moment, second_moment;
    };

public:
    explicit AdamOptimizer(float init_learning_rate,
                            float first_moment_attenuation = 0.9,
                            float second_moment_attenuation = 0.999) :
        Optimizer(init_learning_rate),
        first_moment_attenuation_(first_moment_attenuation),
        second_moment_attenuation_(second_moment_attenuation),
        step_times_(0) {}

    ~AdamOptimizer() override {
        for (auto &p: params_) {
            p.target->release_data(p.first_moment);
            p.target->release_data(p.second_moment);
        }
    }

    void register_layer(Layer &layer) override {
        std::vector<Tensor> layer_params = layer.enum_params();
        for (auto &t: layer_params) {
            params_.emplace_back(
                t,
                t->alloc_data(nullptr),
                t->alloc_data(nullptr)
            );
        }
    }

    void step() override {
        if (step_times_ < (1LL << 40))
            ++step_times_;

        for (auto &p: params_) {
            if (!p.target->grad_data_)
                continue;

            switch (p.target->dtype_) {
            case ScalarType::fp32: {
                const Backend &k = dispatch_kernel(p.target->device_);
                size_t size = p.target->shape_.size;
                Workspace tmp(size * sizeof(float), p.target->device_);
                Data data = p.target->data_, grad = p.target->grad_data_;

                if (step_times_ == 1) {
                    k.mul_scalar_fp32(size, p.first_moment, grad, 1.0f - first_moment_attenuation_);

                    k.square_fp32(size, tmp, grad);
                    k.mul_scalar_fp32(size, p.second_moment, tmp, 1.0f - second_moment_attenuation_);
                }
                else {
                    k.mul_scalar_fp32(size, p.first_moment, p.first_moment, first_moment_attenuation_);
                    k.mul_scalar_fp32(size, tmp, grad, 1.0f - first_moment_attenuation_);
                    k.add_ewise_fp32(size, p.first_moment, p.first_moment, tmp);

                    k.mul_scalar_fp32(size, p.second_moment, p.second_moment, second_moment_attenuation_);
                    k.square_fp32(size, tmp, grad);
                    k.mul_scalar_fp32(size, tmp, tmp, 1.0f - second_moment_attenuation_);
                    k.add_ewise_fp32(size, p.second_moment, p.second_moment, tmp);
                }

                const float first_moment_correction =
                        1.0f / (1.0f - std::pow(first_moment_attenuation_, static_cast<float>(step_times_)));
                const float second_moment_correction =
                        1.0f / (1.0f - std::pow(second_moment_attenuation_, static_cast<float>(step_times_)));

                k.mul_scalar_fp32(size, tmp, p.second_moment, second_moment_correction);
                k.sqrt_fp32(size, tmp, tmp);
                k.add_scalar_fp32(size, tmp, tmp, 1e-8f);
                k.div_ewise_fp32(size, tmp, p.first_moment, tmp);
                k.mul_scalar_fp32(size, tmp, tmp, first_moment_correction * learning_rate_);

                k.sub_ewise_fp32(size, data, data, tmp);
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
            if (!p.target->grad_data_)
                continue;
            p.target->zero_grad();
        }
    }

private:
    std::vector<adam_param> params_;
    float first_moment_attenuation_, second_moment_attenuation_;
    int step_times_;
};
