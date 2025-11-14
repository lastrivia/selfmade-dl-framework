#pragma once

#include <vector>

class TensorImpl;
class Tensor;

class GradNode {
public:
    explicit GradNode(const Tensor &result);

    virtual ~GradNode() = default;

    virtual std::vector<TensorImpl *> inputs() = 0;

    virtual void backward() = 0;

protected:
    TensorImpl *tensor_;
};

static size_t global_no_grad = 0;

class no_grad_lock {
public:
    no_grad_lock() { global_no_grad++; }

    ~no_grad_lock() { global_no_grad--; }
};
