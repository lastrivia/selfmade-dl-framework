#pragma once

#include <cstdint>

namespace kernel_func {

    using scalar_fp32 = void(*)(size_t, char *, float) noexcept;
    using unary = void(*)(size_t, char *, const char *) noexcept;
    using unary_scalar_fp32 = void(*)(size_t, char *, const char *, float) noexcept;
    using unary_tile = void(*)(size_t, size_t, char *, const char *) noexcept;
    using binary = void(*)(size_t, char *, const char *, const char *) noexcept;
    using binary_tile = void(*)(size_t, size_t, char *, const char *, const char *) noexcept;
    using gemm = void(*)(size_t, size_t, size_t, char *, const char *, const char *) noexcept;
    using conv = void(*)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                         char *, const char *, const char *, const char *) noexcept;

}

class kernel {
public:
    kernel_func::binary add_ewise_fp32, sub_ewise_fp32, mul_ewise_fp32, div_ewise_fp32;
    kernel_func::unary_scalar_fp32 add_scalar_fp32, mul_scalar_fp32, pow_fp32;
    kernel_func::scalar_fp32 broadcast_fp32;

    kernel_func::unary square_fp32, sqrt_fp32, relu_fp32;
    kernel_func::binary relu_mask_fp32;

    kernel_func::binary_tile add_cyclic_fp32, sub_cyclic_fp32;
    kernel_func::binary_tile add_stretched_fp32, sub_stretched_fp32;

    kernel_func::unary_tile sum_cyclic_fp32, sum_stretched_fp32, softmax_fp32;

    kernel_func::gemm gemm_fp32[2][2]; // <bool transpose_a, bool transpose_b>
    kernel_func::conv conv_fp32[2][2][2]; // <bool rotate_kernel, bool shift_memory_dim, bool use_bias>

private:
    kernel() = default;
    friend class kernel_init_factory;
};
