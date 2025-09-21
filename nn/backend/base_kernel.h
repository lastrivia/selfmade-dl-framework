#pragma once

#include <cstdint>

enum class device_type: char {
    cpu  = 0,
    cuda = 1
};

enum class data_type: char {
    fp32 = 0
    // todo fp16, ...
};

class tensor;
class tensor_mask;

namespace kernel_func_fp32 {

    using broadcast = void(*)(size_t, float *, float) noexcept;

    using unary = void(*)(size_t, float *, const float *) noexcept;
    using unary_scalar = void(*)(size_t, float *, const float *, float) noexcept;
    using binary = void(*)(size_t, float *, const float *, const float *) noexcept;

    using unary_tile = void(*)(size_t, size_t, float *, const float *) noexcept;
    using binary_tile = void(*)(size_t, size_t, float *, const float *, const float *) noexcept;

    using correct_count = void(*)(size_t, size_t, size_t *, const float *, const float *) noexcept;
    using pool = void(*)(size_t, size_t, size_t, size_t, size_t, float *, bool *, const float *) noexcept;
    using pool_backward = void(*)(size_t, size_t, size_t, size_t, size_t, float *, const bool *, const float *) noexcept;
    using gemm = void(*)(size_t, size_t, size_t, float *, const float *, const float *) noexcept;
    using conv = void(*)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                         float *, const float *, const float *, const float *) noexcept;
    using conv_grad = void(*)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                              float *, const float *, const float *) noexcept;

}

class kernel {
public:
    kernel_func_fp32::binary add_ewise_fp32, sub_ewise_fp32, mul_ewise_fp32, div_ewise_fp32;
    kernel_func_fp32::unary_scalar add_scalar_fp32, mul_scalar_fp32, pow_fp32;

    kernel_func_fp32::broadcast broadcast_fp32;

    kernel_func_fp32::unary square_fp32, sqrt_fp32;

    kernel_func_fp32::unary relu_fp32;
    kernel_func_fp32::binary relu_backward_fp32;

    kernel_func_fp32::binary_tile add_cyclic_fp32, sub_cyclic_fp32;
    kernel_func_fp32::binary_tile add_stretched_fp32, sub_stretched_fp32;
    kernel_func_fp32::unary_tile sum_cyclic_fp32, sum_stretched_fp32;

    kernel_func_fp32::unary_tile softmax_fp32;

    kernel_func_fp32::correct_count correct_count_fp32;

    kernel_func_fp32::pool maxpool_fp32;
    kernel_func_fp32::pool_backward maxpool_backward_fp32;

    kernel_func_fp32::gemm gemm_fp32[2][2]; // <bool transpose_a, bool transpose_b>

    kernel_func_fp32::conv conv_fp32;
    kernel_func_fp32::conv_grad conv_input_grad_fp32, conv_kernel_grad_fp32;

private:
    kernel() = default;
    friend class kernel_init_factory;
};

class any_ptr {
public:
    any_ptr() noexcept : ptr_(nullptr) {}

    any_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {} // NOLINT(*-explicit-constructor)
    any_ptr &operator=(std::nullptr_t) noexcept {
        ptr_ = nullptr;
        return *this;
    }

    any_ptr(void *p) noexcept : ptr_(p) {} // NOLINT(*-explicit-constructor)
    any_ptr &operator=(void *p) noexcept {
        ptr_ = p;
        return *this;
    }

    any_ptr(const any_ptr &) noexcept = default;
    any_ptr &operator=(const any_ptr &) noexcept = default;

    template<typename T>
        requires (std::is_object_v<T> || std::is_void_v<T>)
    operator T *() const noexcept { // NOLINT(*-explicit-constructor)
        return static_cast<T *>(ptr_);
    }

    bool operator==(const any_ptr &) const = default;

private:
    void *ptr_;
};
