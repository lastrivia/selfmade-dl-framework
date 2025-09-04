#pragma once

#include "tensor_interface.h"

inline tensor cross_entropy_grad(const tensor &input_softmax, const tensor &label) {
    return input_softmax - label;
}
