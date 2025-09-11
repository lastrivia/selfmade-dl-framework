#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

#ifdef _MSC_VER
using ssize_t = ptrdiff_t;
#endif

// todo optimize

namespace cpu_kernel {

    template<bool rotate_kernel, bool shift_memory_dim, bool use_bias>
    /**  normal:  [n, c_in, ., .] * [c_out, c_in, ., .] -> [n, c_out, ., .] (forward; input grad)
      *  shifted: [c_in, n, ., .] * [c_in, c_out, ., .] -> [c_out, n, ., .] (weight grad)
      *
      *  === call ===
      *  forward:     <false, false, true>(n, c_in, c_out, h_kernel, w_kernel, h_in, w_in, h_padding, w_padding, out, in, kernel, bias)
      *  input grad:  <true, false, false>(n, c_out, c_in, h_kernel, w_kernel, h_out, w_out,
      *                                    h_kernel - 1 - h_padding, w_kernel - 1 - w_padding, in_grad, out_grad, kernel, <any>)
      *  weight grad: <false, true, false>(c_in, n, c_out, h_out, w_out, h_in, w_in, h_padding, w_padding, w_grad, in, out_grad, <any>)
      *
      *  requires padding < kernel_size
      */
    void conv_fp32(const size_t samples, const size_t c_in, const size_t c_out,
                   const size_t h_kernel, const size_t w_kernel, const size_t h_in, const size_t w_in,
                   const size_t h_padding, const size_t w_padding,
                   char *__restrict dst_p, const char *__restrict in_p, const char *__restrict kernel_p,
                   const char *__restrict bias_p) noexcept {

        const size_t h_out = h_in - h_kernel + 1 + h_padding * 2, w_out = w_in - w_kernel + 1 + w_padding * 2;

        auto in = [=](size_t n, size_t c, size_t h, size_t w) -> const float & {
            return *(reinterpret_cast<const float *__restrict>(in_p) +
                     ((shift_memory_dim ? c * samples + n : n * c_in + c) * h_in + h) * w_in + w);
        };
        auto kernel = [=](size_t co, size_t ci, size_t h, size_t w) -> const float & {
            return *(reinterpret_cast<const float *__restrict>(kernel_p) +
                     ((shift_memory_dim ? ci * c_out + co : co * c_in + ci) * h_kernel + h) * w_kernel + w);
        };
        auto dst = [=](size_t n, size_t c, size_t h, size_t w) -> float & {
            return *(reinterpret_cast<float *__restrict>(dst_p) +
                     ((shift_memory_dim ? c * samples + n : n * c_out + c) * h_out + h) * w_out + w);
        };

        auto bias = reinterpret_cast<const float *__restrict>(bias_p);

        float *tmp = nullptr;
        if (use_bias && bias_p == nullptr) { // not expected
            tmp = new float[c_out];
            memset(tmp, 0, c_out * sizeof(float));
            bias = tmp;
        }

        for (size_t i = 0; i < samples; i++) {
            for (size_t co = 0; co < c_out; co++) {
                for (size_t j = 0; j < h_out; j++) {
                    for (size_t k = 0; k < w_out; k++) {

                        float &dst_e = dst(i, co, j, k);
                        dst_e = use_bias ? bias[co] : 0.0f;
                        size_t h_kernel_start = static_cast<size_t>(std::max(static_cast<ssize_t>(0),
                                                                             static_cast<ssize_t>(h_padding) - static_cast<ssize_t>(j))),
                               h_kernel_end = std::min(h_kernel, h_in + h_padding - j),
                               w_kernel_start = static_cast<size_t>(std::max(static_cast<ssize_t>(0),
                                                                             static_cast<ssize_t>(w_padding) - static_cast<ssize_t>(k))),
                               w_kernel_end = std::min(w_kernel, w_in + w_padding - k);

                        for (size_t ci = 0; ci < c_in; ci++) {
                            for (
                                size_t j_kernel = h_kernel_start;
                                j_kernel < h_kernel_end;
                                j_kernel++
                            ) {
                                for (
                                    size_t k_kernel = w_kernel_start;
                                    k_kernel < w_kernel_end;
                                    k_kernel++
                                ) {
                                    dst_e += in(i, ci, j + j_kernel - h_padding, k + k_kernel - w_padding) *
                                    (rotate_kernel
                                         ? kernel(co, ci, h_kernel - j_kernel - 1, w_kernel - k_kernel - 1)
                                         : kernel(co, ci, j_kernel, k_kernel));
                                }
                            }
                        }

                    }
                }
            }
        }

        delete[] tmp;
    }
}
