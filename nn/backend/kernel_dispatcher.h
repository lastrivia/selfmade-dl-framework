#pragma once

#include "base_kernel.h"
#include "tensor.h"
#include "cpu/conv.h"
#include "cpu/gemm.h"
#include "cpu/ewise.h"

class kernel_init_factory {
    public:
    kernel_init_factory() = delete;

    static kernel init_cpu_kernel() {
        kernel k;

        k.add_ewise_fp32 = cpu_kernel::add_ewise_fp32;
        k.sub_ewise_fp32 = cpu_kernel::sub_ewise_fp32;
        k.mul_ewise_fp32 = cpu_kernel::mul_ewise_fp32;
        k.div_ewise_fp32 = cpu_kernel::div_ewise_fp32;

        k.add_scalar_fp32 = cpu_kernel::add_scalar_fp32;
        k.mul_scalar_fp32 = cpu_kernel::mul_scalar_fp32;
        k.pow_fp32 = cpu_kernel::pow_fp32;

        k.broadcast_fp32 = cpu_kernel::broadcast_fp32;

        k.square_fp32 = cpu_kernel::square_fp32;
        k.sqrt_fp32 = cpu_kernel::sqrt_fp32;

        k.relu_fp32 = cpu_kernel::relu_fp32;
        k.relu_mask_fp32 = cpu_kernel::relu_mask_fp32;

        k.add_cyclic_fp32 = cpu_kernel::add_cyclic_fp32;
        k.sub_cyclic_fp32 = cpu_kernel::sub_cyclic_fp32;
        k.add_stretched_fp32 = cpu_kernel::add_stretched_fp32;
        k.sub_stretched_fp32 = cpu_kernel::sub_stretched_fp32;
        k.sum_cyclic_fp32 = cpu_kernel::sum_cyclic_fp32;
        k.sum_stretched_fp32 = cpu_kernel::sum_stretched_fp32;

        k.softmax_fp32 = cpu_kernel::softmax_fp32;

        k.gemm_fp32[0][0] = cpu_kernel::gemm_fp32<false, false>;
        k.gemm_fp32[0][1] = cpu_kernel::gemm_fp32<false, true>;
        k.gemm_fp32[1][0] = cpu_kernel::gemm_fp32<true, false>;
        k.gemm_fp32[1][1] = cpu_kernel::gemm_fp32<true, true>;

        using kernel_func::conv_mode;
        k.conv_fp32[0] = cpu_kernel::conv_fp32<conv_mode::forward>;
        k.conv_fp32[1] = cpu_kernel::conv_fp32<conv_mode::input_grad>;
        k.conv_fp32[2] = cpu_kernel::conv_fp32<conv_mode::kernel_grad>;

        return k;
    }
};

inline const kernel kernel_dispatch_table[] = {
    kernel_init_factory::init_cpu_kernel() // device_type::cpu == 0
    // todo cuda
};

inline const kernel &dispatch_kernel(const tensor &t) {
    return kernel_dispatch_table[static_cast<size_t>(t.device_type_)];
}
