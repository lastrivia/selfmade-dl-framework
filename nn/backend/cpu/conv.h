#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "../base_kernel.h"

#ifdef _MSC_VER
using ssize_t = ptrdiff_t;
#endif

// todo optimize

namespace cpu_kernel {

    template<kernel_func::conv_mode mode>
    /** === modes ===
     *
     *  forward:
     *
     *  src_a = in, src_b = kernel, dst = out
     *  [n, c_i, h_i, w_i] * [c_o, c_i, h_k, w_k] + bias -> [n, c_o, h_o, w_o]
     *  call: (n, c_i, c_o, h_i, w_i, h_k, w_k, h_pad, w_pad, &out, &in, &kernel, &bias)
     *
     *  input_grad:
     *
     *  src_a = out_grad, src_b = kernel, dst = in_grad
     *  [n, c_o, h_o, w_o] * [c_o, c_i, h_k, w_k](rotated) -> [n, c_i, h_i, w_i]
     *  call: (n, c_i, c_o, h_o, w_o, h_k, w_k, h_k - h_pad - 1, w_k - w_pad - 1, &in_grad, &out_grad, &kernel, nullptr)
     *
     *  kernel_grad:
     *
     *  src_a = in, src_b = out_grad, dst = kernel_grad
     *  [n, c_i, h_i, w_i] * [n, c_o, h_o, w_o] -> [c_o, c_i, h_k, w_k]
     *  call: (n, c_i, c_o, h_i, w_i, h_o, w_o, h_pad, w_pad, &kernel_grad, &in, &out_grad, nullptr)
     *
     *  requires padding < kernel_size
     */
    void conv_fp32(const size_t n, const size_t c_i, const size_t c_o,
                   const size_t h_a, const size_t w_a, const size_t h_b, const size_t w_b,
                   const size_t h_pad, const size_t w_pad,
                   char *__restrict dst_p, const char *__restrict src_a_p, const char *__restrict src_b_p,
                   const char *__restrict bias_p) noexcept {

        const size_t h_dst = h_a - h_b + 1 + h_pad * 2, w_dst = w_a - w_b + 1 + w_pad * 2;

        using enum kernel_func::conv_mode;

        auto src_a = [=](size_t dim_x, size_t dim_y, size_t h, size_t w) -> const float & {
            return *(reinterpret_cast<const float *__restrict>(src_a_p) + ((dim_x * (mode == input_grad ? c_o : c_i) + dim_y) * h_a + h) * w_a + w);
        };
        auto src_b = [=](size_t dim_x, size_t dim_y, size_t h, size_t w) -> const float & {
            return *(reinterpret_cast<const float *__restrict>(src_b_p) + ((dim_x * (mode == kernel_grad ? c_o : c_i) + dim_y) * h_b + h) * w_b + w);
        };
        auto dst = [=](size_t dim_x, size_t dim_y, size_t h, size_t w) -> float & {
            return *(reinterpret_cast<float *__restrict>(dst_p) + ((dim_x * (mode == forward ? c_o : c_i) + dim_y) * h_dst + h) * w_dst + w);
        };

        auto bias = reinterpret_cast<const float *__restrict>(bias_p);

        if (mode == forward && bias_p == nullptr) { // not expected
            return;
        }

        for (size_t dim_i = 0; dim_i < (mode == kernel_grad ? c_o : n); ++dim_i) {
            for (size_t dim_j = 0; dim_j < (mode == forward ? c_o : c_i); ++dim_j) {

                for (size_t i = 0; i < h_dst; ++i) {
                    for (size_t j = 0; j < w_dst; ++j) {

                        float &dst_e = dst(dim_i, dim_j, i, j);
                        dst_e = mode == forward ? bias[dim_j] : 0.0f;

                        size_t b_i_start = static_cast<size_t>(std::max(static_cast<ssize_t>(0),
                                                                          static_cast<ssize_t>(h_pad) - static_cast<ssize_t>(i))),
                               b_i_end = std::min(h_b, h_a + h_pad - i),
                               b_j_start = static_cast<size_t>(std::max(static_cast<ssize_t>(0),
                                                                          static_cast<ssize_t>(w_pad) - static_cast<ssize_t>(j))),
                               b_j_end = std::min(w_b, w_a + w_pad - j);

                        for (size_t dim_k = 0; dim_k < (mode == forward ? c_i : mode == input_grad ? c_o : n); ++dim_k) {
                            for (size_t b_i = b_i_start; b_i < b_i_end; ++b_i) {
                                for (size_t b_j = b_j_start; b_j < b_j_end; ++b_j) {

                                    switch (mode) {
                                    case forward:
                                        dst_e += src_a(dim_i, dim_k, i + b_i - h_pad, j + b_j - w_pad)
                                                * src_b(dim_j, dim_k, b_i, b_j);
                                        break;
                                    case input_grad:
                                        dst_e += src_a(dim_i, dim_k, i + b_i - h_pad, j + b_j - w_pad)
                                                * src_b(dim_k, dim_j, h_b - b_i - 1, w_b - b_j - 1);
                                        break;
                                    case kernel_grad:
                                        dst_e += src_a(dim_k, dim_j, i + b_i - h_pad, j + b_j - w_pad)
                                                * src_b(dim_k, dim_i, b_i, b_j);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
