#pragma once

using kernel_scalar_fp32_fn = void(*)(size_t, char *, float) noexcept;
using kernel_unary_fn = void(*)(size_t, char *, const char *) noexcept;
using kernel_unary_scalar_fp32_fn = void(*)(size_t, char *, const char *, float) noexcept;
using kernel_unary_tile_fn = void(*)(size_t, size_t, char *, const char *) noexcept;
using kernel_binary_fn = void(*)(size_t, char *, const char *, const char *) noexcept;
using kernel_binary_tile_fn = void(*)(size_t, size_t, char *, const char *, const char *) noexcept;
using kernel_gemm_fn = void(*)(size_t, size_t, size_t, char *, const char *, const char *) noexcept;

class kernel {
public:
    kernel(
        kernel_binary_fn add_ewise_fp32_fn, kernel_binary_fn sub_ewise_fp32_fn,
        kernel_binary_fn mul_ewise_fp32_fn, kernel_binary_fn div_ewise_fp32_fn,
        kernel_unary_scalar_fp32_fn add_scalar_fp32_fn,
        kernel_unary_scalar_fp32_fn mul_scalar_fp32_fn, kernel_unary_scalar_fp32_fn pow_fp32_fn,
        kernel_scalar_fp32_fn broadcast_fp32_fn,
        kernel_unary_fn square_fp32_fn, kernel_unary_fn sqrt_fp32_fn,
        kernel_unary_fn relu_fp32_fn, kernel_binary_fn relu_mask_fp32_fn,
        kernel_binary_tile_fn add_cyclic_fp32_fn, kernel_binary_tile_fn sub_cyclic_fp32_fn,
        kernel_binary_tile_fn add_stretched_fp32_fn, kernel_binary_tile_fn sub_stretched_fp32_fn,
        kernel_unary_tile_fn sum_cyclic_fp32_fn, kernel_unary_tile_fn sum_stretched_fp32_fn,
        kernel_unary_tile_fn softmax_fp32_fn,
        kernel_gemm_fn gemm_fp32_fn_nn, kernel_gemm_fn gemm_fp32_fn_nt,
        kernel_gemm_fn gemm_fp32_fn_tn, kernel_gemm_fn gemm_fp32_fn_tt
    ) :
        add_ewise_fp32_(add_ewise_fp32_fn), sub_ewise_fp32_(sub_ewise_fp32_fn),
        mul_ewise_fp32_(mul_ewise_fp32_fn), div_ewise_fp32_(div_ewise_fp32_fn),
        add_scalar_fp32_(add_scalar_fp32_fn), mul_scalar_fp32_(mul_scalar_fp32_fn), pow_fp32_(pow_fp32_fn),
        broadcast_fp32_(broadcast_fp32_fn),
        square_fp32_(square_fp32_fn), sqrt_fp32_(sqrt_fp32_fn),
        relu_fp32_(relu_fp32_fn), relu_mask_fp32_(relu_mask_fp32_fn),
        add_cyclic_fp32_(add_cyclic_fp32_fn), sub_cyclic_fp32_(sub_cyclic_fp32_fn),
        add_stretched_fp32_(add_stretched_fp32_fn), sub_stretched_fp32_(sub_stretched_fp32_fn),
        sum_cyclic_fp32_(sum_cyclic_fp32_fn), sum_stretched_fp32_(sum_stretched_fp32_fn),
        softmax_fp32_(softmax_fp32_fn),
        gemm_fp32_{gemm_fp32_fn_nn, gemm_fp32_fn_nt, gemm_fp32_fn_tn, gemm_fp32_fn_tt} {}

    kernel_binary_fn add_ewise_fp32() const { return add_ewise_fp32_; }
    kernel_binary_fn sub_ewise_fp32() const { return sub_ewise_fp32_; }
    kernel_binary_fn mul_ewise_fp32() const { return mul_ewise_fp32_; }
    kernel_binary_fn div_ewise_fp32() const { return div_ewise_fp32_; }
    kernel_unary_scalar_fp32_fn add_scalar_fp32() const { return add_scalar_fp32_; }
    kernel_unary_scalar_fp32_fn mul_scalar_fp32() const { return mul_scalar_fp32_; }
    kernel_unary_scalar_fp32_fn pow_fp32() const { return pow_fp32_; }
    kernel_scalar_fp32_fn broadcast_fp32() const { return broadcast_fp32_; }
    kernel_unary_fn square_fp32() const { return square_fp32_; }
    kernel_unary_fn sqrt_fp32() const { return sqrt_fp32_; }
    kernel_unary_fn relu_fp32() const { return relu_fp32_; }
    kernel_binary_fn relu_mask_fp32() const { return relu_mask_fp32_; }
    kernel_binary_tile_fn add_cyclic_fp32() const { return add_cyclic_fp32_; }
    kernel_binary_tile_fn sub_cyclic_fp32() const { return sub_cyclic_fp32_; }
    kernel_binary_tile_fn add_stretched_fp32() const { return add_stretched_fp32_; }
    kernel_binary_tile_fn sub_stretched_fp32() const { return sub_stretched_fp32_; }
    kernel_unary_tile_fn sum_cyclic_fp32() const { return sum_cyclic_fp32_; }
    kernel_unary_tile_fn sum_stretched_fp32() const { return sum_stretched_fp32_; }
    kernel_unary_tile_fn softmax_fp32() const { return softmax_fp32_; }

    template<bool transpose_a, bool transpose_b>
    kernel_gemm_fn gemm_fp32() const {
        return gemm_fp32_[transpose_a][transpose_b];
    }

private:
    kernel_binary_fn add_ewise_fp32_, sub_ewise_fp32_, mul_ewise_fp32_, div_ewise_fp32_;
    kernel_unary_scalar_fp32_fn add_scalar_fp32_, mul_scalar_fp32_, pow_fp32_;
    kernel_scalar_fp32_fn broadcast_fp32_;
    kernel_unary_fn square_fp32_, sqrt_fp32_, relu_fp32_;
    kernel_binary_fn relu_mask_fp32_;
    kernel_binary_tile_fn add_cyclic_fp32_, sub_cyclic_fp32_;
    kernel_binary_tile_fn add_stretched_fp32_, sub_stretched_fp32_;
    kernel_unary_tile_fn sum_cyclic_fp32_, sum_stretched_fp32_, softmax_fp32_;
    kernel_gemm_fn gemm_fp32_[2][2];
};
