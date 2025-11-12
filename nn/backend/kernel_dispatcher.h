#pragma once

#include "base_kernel.h"
#include "cpu/operators/common.h"
#include "cpu/operators/conv.h"
#include "cpu/operators/gemm.h"

#ifdef __CUDACC__
#include "cuda/operators/common.cuh"
#include "cuda/operators/conv.cuh"
#include "cuda/operators/gemm.cuh"
#endif

class kernel_init_factory {
public:
    kernel_init_factory() = delete;

    static kernel init_cpu_kernel() {
        kernel k; // NOLINT(*-pro-type-member-init)

        k.copy_fp32 = cpu_kernel::copy_raw<float>;
        k.copy_int32 = cpu_kernel::copy_raw<int32_t>;

        k.add_ewise_fp32 = cpu_kernel::add_ewise<float>;
        k.add_ewise_int32 = cpu_kernel::add_ewise<int32_t>;
        k.sub_ewise_fp32 = cpu_kernel::sub_ewise_fp32;
        k.mul_ewise_fp32 = cpu_kernel::mul_ewise_fp32;
        k.div_ewise_fp32 = cpu_kernel::div_ewise_fp32;

        k.add_scalar_fp32 = cpu_kernel::add_scalar_fp32;
        k.mul_scalar_fp32 = cpu_kernel::mul_scalar_fp32;
        k.pow_fp32 = cpu_kernel::pow_fp32;

        k.broadcast_fp32 = cpu_kernel::broadcast_fp32;
        k.broadcast_int32 = cpu_kernel::broadcast_int32;

        k.square_fp32 = cpu_kernel::square_fp32;
        k.sqrt_fp32 = cpu_kernel::sqrt_fp32;

        k.relu_fp32 = cpu_kernel::relu_fp32;
        k.relu_backward_fp32 = cpu_kernel::relu_backward_fp32;

        k.add_broadcast_fp32 = cpu_kernel::add_broadcast_fp32;
        k.sum_fp32 = cpu_kernel::sum_fp32;

        k.softmax_fp32 = cpu_kernel::softmax_fp32;
        k.log_softmax_fp32 = cpu_kernel::log_softmax_fp32;

        k.correct_count_fp32 = cpu_kernel::correct_count_fp32;

        k.maxpool_fp32 = cpu_kernel::maxpool_fp32;
        k.maxpool_backward_fp32 = cpu_kernel::maxpool_backward_fp32;

        k.gemm_fp32[0][0] = cpu_kernel::gemm_fp32<false, false>;
        k.gemm_fp32[0][1] = cpu_kernel::gemm_fp32<false, true>;
        k.gemm_fp32[1][0] = cpu_kernel::gemm_fp32<true, false>;
        k.gemm_fp32[1][1] = cpu_kernel::gemm_fp32<true, true>;

        k.conv_fp32 = cpu_kernel::conv_fp32;
        k.conv_input_grad_fp32 = cpu_kernel::conv_input_grad_fp32;
        k.conv_kernel_grad_fp32 = cpu_kernel::conv_kernel_grad_fp32;

        return k;
    }

    static kernel init_cuda_kernel() {
        kernel k; // NOLINT(*-pro-type-member-init)

        k.copy_fp32 = cuda_kernel::copy_raw<float>;
        k.copy_int32 = cuda_kernel::copy_raw<int32_t>;

        k.add_ewise_fp32 = cuda_kernel::add_ewise<float>;
        k.add_ewise_int32 = cuda_kernel::add_ewise<int32_t>;
        k.sub_ewise_fp32 = cuda_kernel::sub_ewise_fp32;
        k.mul_ewise_fp32 = cuda_kernel::mul_ewise_fp32;
        k.div_ewise_fp32 = cuda_kernel::div_ewise_fp32;

        k.add_scalar_fp32 = cuda_kernel::add_scalar_fp32;
        k.mul_scalar_fp32 = cuda_kernel::mul_scalar_fp32;
        k.pow_fp32 = cuda_kernel::pow_fp32;

        k.broadcast_fp32 = cuda_kernel::broadcast_fp32;
        k.broadcast_int32 = cuda_kernel::broadcast_int32;

        k.square_fp32 = cuda_kernel::square_fp32;
        k.sqrt_fp32 = cuda_kernel::sqrt_fp32;

        k.relu_fp32 = cuda_kernel::relu_fp32;
        k.relu_backward_fp32 = cuda_kernel::relu_backward_fp32;

        k.add_broadcast_fp32 = cuda_kernel::add_broadcast_fp32;
        k.sum_fp32 = cuda_kernel::sum_fp32;

        k.softmax_fp32 = cuda_kernel::softmax_fp32;
        k.log_softmax_fp32 = cuda_kernel::log_softmax_fp32;

        k.correct_count_fp32 = cuda_kernel::correct_count_fp32;

        k.maxpool_fp32 = cuda_kernel::maxpool_fp32;
        k.maxpool_backward_fp32 = cuda_kernel::maxpool_backward_fp32;

        k.gemm_fp32[0][0] = cuda_kernel::gemm_fp32<false, false>;
        k.gemm_fp32[0][1] = cuda_kernel::gemm_fp32<false, true>;
        k.gemm_fp32[1][0] = cuda_kernel::gemm_fp32<true, false>;
        k.gemm_fp32[1][1] = cuda_kernel::gemm_fp32<true, true>;

        k.conv_fp32 = cuda_kernel::conv_fp32;
        k.conv_input_grad_fp32 = cuda_kernel::conv_input_grad_fp32;
        k.conv_kernel_grad_fp32 = cuda_kernel::conv_kernel_grad_fp32;

        return k;
    }
};

inline const kernel kernel_dispatch_table[] = {
    kernel_init_factory::init_cpu_kernel(), // device_type::cpu  == 0
    kernel_init_factory::init_cuda_kernel() // device_type::cuda == 1
};

inline const kernel &dispatch_kernel(device_desc device) {
    return kernel_dispatch_table[static_cast<size_t>(device.type)];
}
