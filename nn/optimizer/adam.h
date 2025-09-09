#pragma once

#include "base_optimizer.h"

class adam_optimizer : public nn_optimizer {
    struct adam_param {
        tensor &data, &grad;
        tensor first_moment, second_moment;

        explicit adam_param(param target) :
            data(target.data), grad(target.grad),
            first_moment(data.layout()), second_moment(data.layout()) {}
    };

public:
    explicit adam_optimizer(float init_learning_rate,
                            float first_moment_attenuation = 0.9,
                            float second_moment_attenuation = 0.999) :
        nn_optimizer(init_learning_rate),
        first_moment_attenuation_(first_moment_attenuation),
        second_moment_attenuation_(second_moment_attenuation),
        step_times_(0) {}

    ~adam_optimizer() override = default;

    void register_layer(nn_layer &layer) override {
        std::vector<param> layer_params = layer.enum_params();
        for (auto p: layer_params) {
            params_.emplace_back(p);
        }
    }

    void step() override {
        ++step_times_;
        for (auto &p: params_) {
            if (step_times_ == 1) {
                p.first_moment = p.grad * (1.0f - first_moment_attenuation_);
                p.second_moment = square(p.grad) * (1.0f - second_moment_attenuation_);
            }
            else {
                p.first_moment *= first_moment_attenuation_;
                p.first_moment += p.grad * (1.0f - first_moment_attenuation_);
                p.second_moment *= second_moment_attenuation_;
                p.second_moment += square(p.grad) * (1.0f - second_moment_attenuation_);
            }

            const float first_moment_correction = 1.0f / (1.0f - std::pow(first_moment_attenuation_,
                                                                          static_cast<float>(step_times_)));
            const float second_moment_correction = 1.0f / (1.0f - std::pow(second_moment_attenuation_,
                                                                           static_cast<float>(step_times_)));

            p.data -= div_ewise(
                p.first_moment * (first_moment_correction * learning_rate_),
                sqrt(p.second_moment * second_moment_correction) + 1e-8f
            );
        }
    }

private:
    std::vector<adam_param> params_;
    float first_moment_attenuation_, second_moment_attenuation_;
    int step_times_;
};
