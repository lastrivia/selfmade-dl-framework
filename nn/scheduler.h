#pragma once

#include "optimizer/base_optimizer.h"

class nn_scheduler {
public:
    nn_scheduler() : optimizer_(nullptr) {}

    virtual ~nn_scheduler() = default;

    virtual void bind_optimizer(nn_optimizer *optimizer) = 0;

    float &learning_rate() {
        return optimizer_->learning_rate_;
    }

protected:
    nn_optimizer *optimizer_;
};

class exponential_scheduler : public nn_scheduler {
public:
    explicit exponential_scheduler(float scalar) : scalar_(scalar) {}

    ~exponential_scheduler() override = default;

    void bind_optimizer(nn_optimizer *optimizer) override {
        optimizer_ = optimizer;
    }

    void step() {
        learning_rate() *= scalar_;
    }

private:
    float scalar_;
};
