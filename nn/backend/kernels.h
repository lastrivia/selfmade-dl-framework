#pragma once

#include "base_kernel.h"
#include "tensor.h"
#include "cpu/gemm.h"
#include "cpu/ewise.h"

inline kernel cpu_kernel = {
    cpu_kernel_add_ewise_fp32,
    cpu_kernel_sub_ewise_fp32,
    cpu_kernel_mul_ewise_fp32,
    cpu_kernel_div_ewise_fp32,
    cpu_kernel_add_scalar_fp32,
    cpu_kernel_mul_scalar_fp32,
    cpu_kernel_pow_fp32,
    cpu_kernel_broadcast_fp32,
    cpu_kernel_square_fp32,
    cpu_kernel_sqrt_fp32,
    cpu_kernel_relu_fp32,
    cpu_kernel_relu_mask_fp32,
    cpu_kernel_add_cyclic_fp32,
    cpu_kernel_sub_cyclic_fp32,
    cpu_kernel_add_stretched_fp32,
    cpu_kernel_sub_stretched_fp32,
    cpu_kernel_sum_cyclic_fp32,
    cpu_kernel_sum_stretched_fp32,
    cpu_kernel_softmax_fp32,
    cpu_kernel_gemm_fp32<false, false>,
    cpu_kernel_gemm_fp32<false, true>,
    cpu_kernel_gemm_fp32<true, false>,
    cpu_kernel_gemm_fp32<true, true>
};

constexpr const kernel *device_kernel_table[] = {
    &cpu_kernel // device_type::cpu == 0
    // todo cuda
};

inline const kernel *get_kernel(const tensor &t) {
    return device_kernel_table[static_cast<size_t>(t.device_type_)];
}
