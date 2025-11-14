#pragma once

#include "layer/base_layer.h"
#include "model.h"

class Optimizer {
public:
    explicit Optimizer(float init_learning_rate) : learning_rate_(init_learning_rate) {}

    virtual ~Optimizer() = default;

    virtual void register_layer(Layer &layer) = 0;

    virtual void register_model(Model &model) {
        for (auto &layer : model.layers_)
            register_layer(*layer);
    }

    virtual void step() = 0;

    virtual void zero_grad() = 0;

    friend class Scheduler;

protected:
    float learning_rate_;
};
