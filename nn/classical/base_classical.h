#pragma once

#include "tensor.h"

class BaseClassicalAlgorithm {
public:
    BaseClassicalAlgorithm() = default;

    virtual ~BaseClassicalAlgorithm() = default;

    virtual void fit(Tensor features, Tensor labels) = 0;

    virtual Tensor predict(const Tensor &features) = 0;
};
