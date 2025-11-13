#pragma once

namespace cpu_backend {

    static constexpr size_t NDIM_STACK_BUF_SIZE = 8;

    inline size_t calc_strides(size_t *strides, size_t ndim, const size_t *lengths, const bool *mask) {
        size_t x = 1;
        for (size_t i = 0; i < ndim; i++) {
            if (mask[i]) {
                strides[i] = x;
                x *= lengths[i];
            }
            else strides[i] = 0;
        }
        return x;
    }

}
