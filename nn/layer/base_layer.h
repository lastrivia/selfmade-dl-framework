#pragma once

#include "../backend.h"

struct param {
    tensor &data;
    tensor &grad;

    param(tensor &data, tensor &grad): data(data), grad(grad) {}
};

class nn_layer {
public:
    virtual ~nn_layer() = default;

    virtual tensor forward_propagation(const tensor &activation) = 0;

    virtual tensor back_propagation(const tensor &gradient) = 0;

    virtual std::vector<param> enum_params() = 0;
};
