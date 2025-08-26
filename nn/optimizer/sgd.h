#pragma once

#include "base_optimizer.h"

class sgd_optimizer : public nn_optimizer {
public:
    explicit sgd_optimizer(float init_learning_rate) : nn_optimizer(init_learning_rate) {}

    ~sgd_optimizer() override = default;

    void register_layer(nn_layer *layer) override {
        std::vector<param *> layer_params = layer->enum_params();
        for (auto p: layer_params) {
            params_.push_back(p);
        }
    }

    void step() override {
        for (auto p: params_) {
            p->data -= p->grad * learning_rate_;
        }
    }

private:
    std::vector<param *> params_;
};
