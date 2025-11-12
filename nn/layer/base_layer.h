#pragma once

#include "../backend.h"

class nn_layer {
public:
    virtual ~nn_layer() = default;

    virtual tensor operator()(const tensor &x) = 0;

    virtual std::vector<tensor> enum_params() = 0;

    virtual void to_device(device_desc device) = 0;
};
