#pragma once

#include <cstdint>

#include "except.h"

enum class device_type: char {
    cpu  = 0,
    cuda = 1
};

class device_desc {
public:
    device_type type;
    // todo device_id; support multi-cards

    device_desc() : type(device_type::cpu) {}

    device_desc(device_type type) : type(type) {} // NOLINT(*-explicit-constructor)

    device_desc(const device_desc &other) = default;

    device_desc(const char *device_name) { // NOLINT(*-explicit-constructor)
        if (device_name == nullptr)
            throw nn_except("device name not provided", __FILE__, __LINE__);
        if (strcmp("cpu", device_name) == 0)
            type = device_type::cpu;
        else if (strcmp("cuda", device_name) == 0)
            type = device_type::cuda;
        else
            throw nn_except("unknown device name", __FILE__, __LINE__);
    }

    bool operator==(const device_desc &other) const = default;
    bool operator!=(const device_desc &other) const = default;
};

enum class data_type: char {
    fp32  = 0,
    int32 = 1 // unused, just for testing unsupported cases in code generation
};

inline size_t size_of(data_type type) {
    switch (type) {
    case data_type::fp32:
        return sizeof(float);
    case data_type::int32:
        return sizeof(int32_t);
    }
    return 0;
}

namespace kernel_func {

    // todo replace with this:
    // template<typename T>
    // using broadcast = void(*)(size_t, T *, T);

    namespace fp32 {

        using broadcast = void(*)(size_t, float *, float);

        using unary = void(*)(size_t, float *, const float *);
        using unary_scalar = void(*)(size_t, float *, const float *, float);
        using binary = void(*)(size_t, float *, const float *, const float *);

        using unary_tile = void(*)(size_t, size_t, float *, const float *);
        using binary_tile = void(*)(size_t, size_t, float *, const float *, const float *); // abandoned

        using unary_ndim = void(*)(size_t, size_t, const size_t *, const bool *, float *, const float *);
        using binary_ndim = void(*)(size_t, size_t, const size_t *, const bool *, const bool *, float *, const float *, const float *);

        using correct_count = void(*)(size_t, size_t, size_t *, const float *, const float *);
        using pool = void(*)(size_t, size_t, size_t, size_t, size_t, float *, bool *, const float *);
        using pool_backward = void(*)(size_t, size_t, size_t, size_t, size_t, float *, const bool *, const float *);

        using gemm = void(*)(size_t, size_t, size_t, float *, const float *, const float *);

        using conv = void(*)(size_t, size_t, size_t,
                             const float *, size_t, size_t,
                             const float *, size_t, size_t,
                             size_t, size_t, const float *, float *);

        using conv_grad = void(*)(size_t, size_t, size_t,
                                  const float *, size_t, size_t,
                                  const float *, size_t, size_t,
                                  size_t, size_t, float *);

    }

    namespace int32 {

        using broadcast = void(*)(size_t, int32_t *, int32_t);
        using unary = void(*)(size_t, int32_t *, const int32_t *);
        using binary = void(*)(size_t, int32_t *, const int32_t *, const int32_t *);
    }
}

class backend {
public:
    // fp32
    kernel_func::fp32::unary copy_fp32;
    kernel_func::fp32::broadcast broadcast_fp32;

    kernel_func::fp32::binary add_ewise_fp32, sub_ewise_fp32, mul_ewise_fp32, div_ewise_fp32;
    kernel_func::fp32::unary_scalar add_scalar_fp32, mul_scalar_fp32, pow_fp32;


    kernel_func::fp32::unary square_fp32, sqrt_fp32;

    kernel_func::fp32::unary relu_fp32;
    kernel_func::fp32::binary relu_backward_fp32;

    kernel_func::fp32::binary_ndim add_broadcast_fp32;
    kernel_func::fp32::unary_ndim sum_fp32;

    kernel_func::fp32::unary_tile softmax_fp32, log_softmax_fp32;

    kernel_func::fp32::correct_count correct_count_fp32;

    kernel_func::fp32::pool maxpool_fp32;
    kernel_func::fp32::pool_backward maxpool_backward_fp32;

    kernel_func::fp32::gemm gemm_fp32[2][2]; // <bool transpose_a, bool transpose_b>

    kernel_func::fp32::conv conv_fp32;
    kernel_func::fp32::conv_grad conv_input_grad_fp32, conv_kernel_grad_fp32;

    // int32
    kernel_func::int32::unary copy_int32;
    kernel_func::int32::broadcast broadcast_int32;

    kernel_func::int32::binary add_ewise_int32;

private:
    backend() = default;
    friend class backend_init_factory;
};

class data_ptr {
public:
    data_ptr() noexcept : ptr_(nullptr) {}

    data_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {} // NOLINT(*-explicit-constructor)
    data_ptr &operator=(std::nullptr_t) noexcept {
        ptr_ = nullptr;
        return *this;
    }

    data_ptr(void *p) noexcept : ptr_(p) {} // NOLINT(*-explicit-constructor)
    data_ptr &operator=(void *p) noexcept {
        ptr_ = p;
        return *this;
    }

    data_ptr(const data_ptr &) noexcept = default;
    data_ptr &operator=(const data_ptr &) noexcept = default;

    template<typename T>
        requires (std::is_object_v<T> || std::is_void_v<T>)
    operator T *() const noexcept { // NOLINT(*-explicit-constructor)
        return static_cast<T *>(ptr_);
    }

    explicit operator bool() const noexcept {
        return ptr_ != nullptr;
    }

    bool operator==(const data_ptr &) const = default;

private:
    void *ptr_;
};
