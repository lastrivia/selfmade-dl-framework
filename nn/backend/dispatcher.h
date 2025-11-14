#pragma once

#include "base_backend.h"
#include "cpu/operators/common.h"
#include "cpu/operators/conv.h"
#include "cpu/operators/gemm.h"

#ifdef __CUDACC__
#include "cuda/operators/common.cuh"
#include "cuda/operators/conv.cuh"
#include "cuda/operators/gemm.cuh"
#endif

class BackendInitFactory {
public:
    BackendInitFactory() = delete;

    static Backend init_cpu_backend() {
        Backend k; // NOLINT(*-pro-type-member-init)

        k.copy_fp32 = cpu_backend::copy_raw<float>;
        k.copy_int32 = cpu_backend::copy_raw<int32_t>;

        k.add_ewise_fp32 = cpu_backend::add_ewise<float>;
        k.add_ewise_int32 = cpu_backend::add_ewise<int32_t>;
        k.sub_ewise_fp32 = cpu_backend::sub_ewise_fp32;
        k.mul_ewise_fp32 = cpu_backend::mul_ewise_fp32;
        k.div_ewise_fp32 = cpu_backend::div_ewise_fp32;

        k.add_scalar_fp32 = cpu_backend::add_scalar_fp32;
        k.mul_scalar_fp32 = cpu_backend::mul_scalar_fp32;
        k.pow_fp32 = cpu_backend::pow_fp32;

        k.broadcast_fp32 = cpu_backend::broadcast_fp32;
        k.broadcast_int32 = cpu_backend::broadcast_int32;

        k.square_fp32 = cpu_backend::square_fp32;
        k.sqrt_fp32 = cpu_backend::sqrt_fp32;

        k.relu_fp32 = cpu_backend::relu_fp32;
        k.relu_backward_fp32 = cpu_backend::relu_backward_fp32;

        k.add_broadcast_fp32 = cpu_backend::add_broadcast_fp32;
        k.sum_fp32 = cpu_backend::sum_fp32;

        k.softmax_fp32 = cpu_backend::softmax_fp32;
        k.log_softmax_fp32 = cpu_backend::log_softmax_fp32;

        k.correct_count_fp32 = cpu_backend::correct_count_fp32;

        k.maxpool_fp32 = cpu_backend::maxpool_fp32;
        k.maxpool_backward_fp32 = cpu_backend::maxpool_backward_fp32;

        k.gemm_fp32[0][0] = cpu_backend::gemm_fp32<false, false>;
        k.gemm_fp32[0][1] = cpu_backend::gemm_fp32<false, true>;
        k.gemm_fp32[1][0] = cpu_backend::gemm_fp32<true, false>;
        k.gemm_fp32[1][1] = cpu_backend::gemm_fp32<true, true>;

        k.conv_fp32 = cpu_backend::conv_fp32;
        k.conv_input_grad_fp32 = cpu_backend::conv_input_grad_fp32;
        k.conv_kernel_grad_fp32 = cpu_backend::conv_kernel_grad_fp32;

        return k;
    }

    static Backend init_cuda_backend() {
        Backend k; // NOLINT(*-pro-type-member-init)

        k.copy_fp32 = cuda_backend::copy_raw<float>;
        k.copy_int32 = cuda_backend::copy_raw<int32_t>;

        k.add_ewise_fp32 = cuda_backend::add_ewise<float>;
        k.add_ewise_int32 = cuda_backend::add_ewise<int32_t>;
        k.sub_ewise_fp32 = cuda_backend::sub_ewise_fp32;
        k.mul_ewise_fp32 = cuda_backend::mul_ewise_fp32;
        k.div_ewise_fp32 = cuda_backend::div_ewise_fp32;

        k.add_scalar_fp32 = cuda_backend::add_scalar_fp32;
        k.mul_scalar_fp32 = cuda_backend::mul_scalar_fp32;
        k.pow_fp32 = cuda_backend::pow_fp32;

        k.broadcast_fp32 = cuda_backend::broadcast_fp32;
        k.broadcast_int32 = cuda_backend::broadcast_int32;

        k.square_fp32 = cuda_backend::square_fp32;
        k.sqrt_fp32 = cuda_backend::sqrt_fp32;

        k.relu_fp32 = cuda_backend::relu_fp32;
        k.relu_backward_fp32 = cuda_backend::relu_backward_fp32;

        k.add_broadcast_fp32 = cuda_backend::add_broadcast_fp32;
        k.sum_fp32 = cuda_backend::sum_fp32;

        k.softmax_fp32 = cuda_backend::softmax_fp32;
        k.log_softmax_fp32 = cuda_backend::log_softmax_fp32;

        k.correct_count_fp32 = cuda_backend::correct_count_fp32;

        k.maxpool_fp32 = cuda_backend::maxpool_fp32;
        k.maxpool_backward_fp32 = cuda_backend::maxpool_backward_fp32;

        k.gemm_fp32[0][0] = cuda_backend::gemm_fp32<false, false>;
        k.gemm_fp32[0][1] = cuda_backend::gemm_fp32<false, true>;
        k.gemm_fp32[1][0] = cuda_backend::gemm_fp32<true, false>;
        k.gemm_fp32[1][1] = cuda_backend::gemm_fp32<true, true>;

        k.conv_fp32 = cuda_backend::conv_fp32;
        k.conv_input_grad_fp32 = cuda_backend::conv_input_grad_fp32;
        k.conv_kernel_grad_fp32 = cuda_backend::conv_kernel_grad_fp32;

        return k;
    }
};

inline const Backend backend_dispatch_table[] = {
    BackendInitFactory::init_cpu_backend(), // device_type::cpu  == 0
    BackendInitFactory::init_cuda_backend() // device_type::cuda == 1
};

inline const Backend &dispatch_kernel(DeviceDesc device) {
    return backend_dispatch_table[static_cast<size_t>(device.type)];
}
