#pragma once

#include "../backend.h"

class param {
public:
    tensor data;
    tensor grad;
    int reg_index;

    explicit param(const int rows): data(rows), grad(rows), reg_index(-1) {}

    param(const int rows, const int cols): data(rows, cols), grad(rows, cols), reg_index(-1) {}

    int rows() const {
        return data.rows();
    }

    int cols() const {
        return data.cols();
    }
};

class nn_layer {
public:
    virtual ~nn_layer() = default;

    virtual tensor forward_propagation(const tensor &activation) = 0;

    virtual tensor back_propagation(const tensor &gradient) = 0;

    virtual std::vector<param *> enum_params() = 0;
};
