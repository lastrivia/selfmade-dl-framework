#pragma once

#include <vector>

class tensor_impl;
class tensor;

class grad_node {
public:
    grad_node(const tensor &result);

    virtual ~grad_node() {}

    virtual std::vector<tensor_impl *> inputs() = 0;

    virtual void backward() = 0;

protected:
    tensor_impl *tensor_;
};

static size_t global_no_grad = 0;

class no_grad_lock {
public:
    no_grad_lock() { global_no_grad++; }

    ~no_grad_lock() { global_no_grad--; }
};
