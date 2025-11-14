#pragma once

#include "tensor.h"

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor operator()(const Tensor &x) = 0;

    virtual std::vector<Tensor> enum_params() = 0;

    virtual void to_device(DeviceDesc device) = 0;
};
