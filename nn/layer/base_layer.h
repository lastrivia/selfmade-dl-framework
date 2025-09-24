#pragma once

#include "../backend.h"

struct param {
    tensor &data;
    tensor &grad;

    param(tensor &data, tensor &grad) : data(data), grad(grad) {}
};

class nn_layer {
public:
    virtual ~nn_layer() = default;

    virtual tensor forward_propagation(tensor &activation) = 0;

    virtual tensor back_propagation(tensor &gradient) = 0;

    virtual std::vector<param> enum_params() = 0;

    void to_device(device_type_arg device) {
        for (auto &p: enum_params()) {
            p.data.to_device(device);
            p.grad.to_device(device);
        }
    }
};
