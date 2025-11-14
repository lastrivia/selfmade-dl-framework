#pragma once

#include "optimizer/base_optimizer.h"

class Scheduler {
public:
    Scheduler() : optimizer_(nullptr) {}

    virtual ~Scheduler() = default;

    virtual void bind_optimizer(Optimizer *optimizer) = 0;

    float &learning_rate() {
        return optimizer_->learning_rate_;
    }

protected:
    Optimizer *optimizer_;
};

class ExponentialScheduler : public Scheduler {
public:
    explicit ExponentialScheduler(float scalar) : scalar_(scalar) {}

    ~ExponentialScheduler() override = default;

    void bind_optimizer(Optimizer *optimizer) override {
        optimizer_ = optimizer;
    }

    void step() {
        learning_rate() *= scalar_;
    }

private:
    float scalar_;
};
