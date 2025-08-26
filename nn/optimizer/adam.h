#pragma once

#include "base_optimizer.h"

class adam_optimizer : public nn_optimizer {
    struct param_data {
        param *target;
        tensor first_moment, second_moment;

        explicit param_data(param *target_param) :
            target(target_param),
            first_moment(target_param->rows(), target_param->cols()),
            second_moment(target_param->rows(), target_param->cols()) {}
    };

public:
    explicit adam_optimizer(float init_learning_rate,
                            float first_moment_attenuation = 0.9,
                            float second_moment_attenuation = 0.999):
        nn_optimizer(init_learning_rate),
        first_moment_attenuation_(first_moment_attenuation),
        second_moment_attenuation_(second_moment_attenuation),
        step_times_(0) {}

    ~adam_optimizer() override = default;

    void register_layer(nn_layer *layer) override {
        std::vector<param *> layer_params = layer->enum_params();
        for (auto p: layer_params) {
            params_data_.emplace_back(p);
        }
    }

    void step() override {
        ++step_times_;
        for (auto &p: params_data_) {
            if (step_times_ == 1) {
                p.first_moment = p.target->grad * (1.0f - first_moment_attenuation_);
                p.second_moment = per_element_sqr(p.target->grad) * (1.0f - second_moment_attenuation_);
            } else {
                p.first_moment *= first_moment_attenuation_;
                p.first_moment += p.target->grad * (1.0f - first_moment_attenuation_);
                p.second_moment *= second_moment_attenuation_;
                p.second_moment += per_element_sqr(p.target->grad) * (1.0f - second_moment_attenuation_);
            }

            const float first_moment_correction = 1.0f - std::pow(first_moment_attenuation_, static_cast<float>(step_times_));
            const float second_moment_correction = 1.0f - std::pow(second_moment_attenuation_, static_cast<float>(step_times_));

            for (int i = 0; i < p.target->rows(); ++i) {
                for (int j = 0; j < p.target->cols(); ++j) {
                    p.target->data(i, j) -= learning_rate_ * (p.first_moment(i, j) / first_moment_correction)
                            / (std::sqrt(p.second_moment(i, j) / second_moment_correction) + 1e-8f);
                }
            }
        }
    }

private:
    std::vector<param_data> params_data_;
    float first_moment_attenuation_, second_moment_attenuation_;
    int step_times_;
};
