#pragma once

#include "../layer/base_layer.h"
#include "../base_model.h"

class nn_optimizer {
public:
    explicit nn_optimizer(float init_learning_rate) : learning_rate_(init_learning_rate) {}

    virtual ~nn_optimizer() = default;

    virtual void register_layer(nn_layer &layer) = 0;

    virtual void register_model(nn_model &model) {
        for (auto &layer : model.layers_)
            register_layer(*layer);
    }

    virtual void step() = 0;

    friend class nn_scheduler;

protected:
    float learning_rate_;
};
