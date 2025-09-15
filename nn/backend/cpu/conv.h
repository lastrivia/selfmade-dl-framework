#pragma once

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "constants.h"
#include "../base_kernel.h"

#ifdef _MSC_VER
using ssize_t = ptrdiff_t;
#endif

// todo optimize

namespace cpu_kernel {
    /** === modes ===
     *
     *  forward:
     *
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

    namespace conv_utils_fp32 {

        using enum kernel_func::conv_mode;

        /** === FORWARD ===
         *
         *  in * kernel + bias -> out
         *  [n, c_i, h_i, w_i] * [c_o, c_i, h_k, w_k] + bias -> [n, c_o, h_o, w_o]
         *
         *  multithreading over: n
         *  vectorize: w_i (code: w_in) & w_o (code: w_dst)
         */
        inline void conv_fp32_worker(const size_t c_i, const size_t c_o,
                                     const size_t mt_begin, const size_t mt_end,
                                     const ssize_t h_in, const ssize_t w_in, const ssize_t h_ker, const ssize_t w_ker,
                                     const ssize_t h_pad, const ssize_t w_pad,
                                     float *__restrict dst, const float *__restrict in, const float *__restrict ker,
                                     const float *__restrict bias) noexcept {

            const ssize_t h_dst = h_in - h_ker + 1 + h_pad * 2, w_dst = w_in - w_ker + 1 + w_pad * 2;

            const ssize_t dst_stride[2] = {static_cast<ssize_t>(c_o) * h_dst * w_dst, h_dst * w_dst},
                          in_stride[2] = {static_cast<ssize_t>(c_i) * h_in * w_in, h_in * w_in},
                          ker_stride[2] = {static_cast<ssize_t>(c_i) * h_ker * w_ker, h_ker * w_ker};

            for (size_t dim_i = mt_begin; dim_i < mt_end; ++dim_i) { // n
                for (size_t dim_j = 0; dim_j < c_o; ++dim_j) {
                    float *dst_loc = dst + dim_i * dst_stride[0] + dim_j * dst_stride[1]; // local offsets

                    for (ssize_t i_dst = 0; i_dst < h_dst; ++i_dst) {

                        // i_in = i_dst + i_ker - h_pad \in [0, h_in)
                        const ssize_t i_ker_start = std::max(static_cast<ssize_t>(0), h_pad - i_dst),
                                      i_ker_end = std::min(h_ker, h_in + h_pad - i_dst);

                        ssize_t j_dst = 0;

                        // === AVX2 PART ===
                        for (; j_dst + AVX2_FP32_N <= w_dst; j_dst += AVX2_FP32_N) {

                            // j_in[] = j_dst[] + j_ker - w_pad \in [0, w_in)
                            const ssize_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst - static_cast<ssize_t>(AVX2_FP32_N) + 1),
                                          j_ker_loadu_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                          j_ker_loadu_end = std::min(w_ker, w_in + w_pad - j_dst - static_cast<ssize_t>(AVX2_FP32_N) + 1),
                                          j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            __m256 dst_vec = _mm256_set1_ps(bias[dim_j]);

                            for (size_t dim_k = 0; dim_k < c_i; ++dim_k) {
                                const float *in_loc = in + dim_i * in_stride[0] + dim_k * in_stride[1] + (i_dst - h_pad) * w_in,
                                            *ker_loc = ker + dim_j * ker_stride[0] + dim_k * ker_stride[1];

                                for (ssize_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker) {

                                    auto boundary_load_in = [&](ssize_t j_in) {
                                        const float *in_row = in_loc + i_ker * w_in;
                                        alignas(32) float in_tmp[AVX2_FP32_N] = {0.0f};
                                        for (ssize_t offset = 0; offset < static_cast<ssize_t>(AVX2_FP32_N); ++offset) {
                                            if (j_in + offset >= static_cast<ssize_t>(0) && j_in + offset < w_in)
                                                in_tmp[offset] = in_row[j_in + offset];
                                        }
                                        return _mm256_load_ps(in_tmp);
                                        // I've tried other ways but this seems to be the fastest?
                                    };

                                    ssize_t j_ker = j_ker_start;
                                    for (; j_ker < j_ker_loadu_start; ++j_ker)
                                        dst_vec = _mm256_fmadd_ps(
                                            _mm256_set1_ps(ker_loc[i_ker * w_ker + j_ker]),
                                            boundary_load_in(j_dst + j_ker - w_pad),
                                            dst_vec
                                        );
                                    for (; j_ker < j_ker_loadu_end; ++j_ker)
                                        dst_vec = _mm256_fmadd_ps(
                                            _mm256_set1_ps(ker_loc[i_ker * w_ker + j_ker]),
                                            _mm256_loadu_ps(in_loc + i_ker * w_in + (j_dst + j_ker - w_pad)),
                                            dst_vec
                                        );
                                    for (; j_ker < j_ker_end; ++j_ker)
                                        dst_vec = _mm256_fmadd_ps(
                                            _mm256_set1_ps(ker_loc[i_ker * w_ker + j_ker]),
                                            boundary_load_in(j_dst + j_ker - w_pad),
                                            dst_vec
                                        );
                                }
                            }
                            _mm256_storeu_ps(dst_loc + i_dst * w_dst + j_dst, dst_vec);
                        }

                        // === SCALAR PART ===
                        for (; j_dst < w_dst; ++j_dst) {
                            float dst_e = bias[dim_j];

                            size_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                   j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            for (size_t dim_k = 0; dim_k < c_i; ++dim_k) {
                                const float *in_loc = in + dim_i * in_stride[0] + dim_k * in_stride[1] + (i_dst - h_pad) * w_in,
                                            *ker_loc = ker + dim_j * ker_stride[0] + dim_k * ker_stride[1];

                                for (size_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker) {
                                    for (size_t j_ker = j_ker_start; j_ker < j_ker_end; ++j_ker) {
                                        dst_e += in_loc[i_ker * w_in + (j_dst + j_ker - w_pad)] * ker_loc[i_ker * w_ker + j_ker];
                                    }
                                }
                            }
                            dst_loc[i_dst * w_dst + j_dst] = dst_e;
                        }
                    }
                }
            }
        }


        /** === INPUT GRAD ===
         *
         *  out_grad * kernel(rotated) -> in_grad
         *  [n, c_o, h_o, w_o] * [c_o, c_i, h_k, w_k](rotated) -> [n, c_i, h_i, w_i]
         *
         *  multithreading over: n
         *  vectorize: w_i (code: w_dst) & w_o (code: w_in)
         *
         *  here h_pad = h_ker - (forward)h_pad - 1
         */
        inline void conv_input_grad_fp32_worker(const size_t c_i, const size_t c_o,
                                                const size_t mt_begin, const size_t mt_end,
                                                const ssize_t h_in, const ssize_t w_in, const ssize_t h_ker, const ssize_t w_ker,
                                                const ssize_t h_pad, const ssize_t w_pad,
                                                float *__restrict dst, const float *__restrict in,
                                                const float *__restrict ker) noexcept {

            const ssize_t h_dst = h_in - h_ker + 1 + h_pad * 2, w_dst = w_in - w_ker + 1 + w_pad * 2;

            const ssize_t dst_stride[2] = {static_cast<ssize_t>(c_i) * h_dst * w_dst, h_dst * w_dst},
                          in_stride[2] = {static_cast<ssize_t>(c_o) * h_in * w_in, h_in * w_in},
                          ker_stride[2] = {static_cast<ssize_t>(c_i) * h_ker * w_ker, h_ker * w_ker};

            for (size_t dim_i = mt_begin; dim_i < mt_end; ++dim_i) { // n
                for (size_t dim_j = 0; dim_j < c_i; ++dim_j) {
                    float *dst_loc = dst + dim_i * dst_stride[0] + dim_j * dst_stride[1]; // local offsets

                    for (ssize_t i_dst = 0; i_dst < h_dst; ++i_dst) {

                        // i_in = i_dst + i_ker - h_pad \in [0, h_in)
                        const ssize_t i_ker_start = std::max(static_cast<ssize_t>(0), h_pad - i_dst),
                                      i_ker_end = std::min(h_ker, h_in + h_pad - i_dst);

                        ssize_t j_dst = 0;

                        // === AVX2 PART ===
                        for (; j_dst + AVX2_FP32_N <= w_dst; j_dst += AVX2_FP32_N) {

                            // j_in[] = j_dst[] + j_ker - w_pad \in [0, w_in)
                            const ssize_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst - static_cast<ssize_t>(AVX2_FP32_N) + 1),
                                          j_ker_loadu_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                          j_ker_loadu_end = std::min(w_ker, w_in + w_pad - j_dst - static_cast<ssize_t>(AVX2_FP32_N) + 1),
                                          j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            __m256 dst_vec = _mm256_setzero_ps();

                            for (size_t dim_k = 0; dim_k < c_o; ++dim_k) {
                                const float *in_loc = in + dim_i * in_stride[0] + dim_k * in_stride[1] + (i_dst - h_pad) * w_in,
                                            *ker_loc = ker + dim_k * ker_stride[0] + dim_j * ker_stride[1] + ker_stride[1] - 1;

                                for (ssize_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker) {

                                    auto boundary_load_in = [&](ssize_t j_in) {
                                        const float *in_row = in_loc + i_ker * w_in;
                                        alignas(32) float in_tmp[AVX2_FP32_N] = {0.0f};
                                        for (ssize_t offset = 0; offset < static_cast<ssize_t>(AVX2_FP32_N); ++offset) {
                                            if (j_in + offset >= static_cast<ssize_t>(0) && j_in + offset < w_in)
                                                in_tmp[offset] = in_row[j_in + offset];
                                        }
                                        return _mm256_load_ps(in_tmp);
                                    };

                                    ssize_t j_ker = j_ker_start;
                                    for (; j_ker < j_ker_loadu_start; ++j_ker)
                                        dst_vec = _mm256_fmadd_ps(
                                            _mm256_set1_ps(ker_loc[-i_ker * w_ker - j_ker]),
                                            boundary_load_in(j_dst + j_ker - w_pad),
                                            dst_vec
                                        );
                                    for (; j_ker < j_ker_loadu_end; ++j_ker)
                                        dst_vec = _mm256_fmadd_ps(
                                            _mm256_set1_ps(ker_loc[-i_ker * w_ker - j_ker]),
                                            _mm256_loadu_ps(in_loc + i_ker * w_in + (j_dst + j_ker - w_pad)),
                                            dst_vec
                                        );
                                    for (; j_ker < j_ker_end; ++j_ker)
                                        dst_vec = _mm256_fmadd_ps(
                                            _mm256_set1_ps(ker_loc[-i_ker * w_ker - j_ker]),
                                            boundary_load_in(j_dst + j_ker - w_pad),
                                            dst_vec
                                        );
                                }
                            }
                            _mm256_storeu_ps(dst_loc + i_dst * w_dst + j_dst, dst_vec);
                        }

                        // === SCALAR PART ===
                        for (; j_dst < w_dst; ++j_dst) {
                            float dst_e = 0.0f;

                            size_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                   j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            for (size_t dim_k = 0; dim_k < c_o; ++dim_k) {
                                const float *in_loc = in + dim_i * in_stride[0] + dim_k * in_stride[1] + (i_dst - h_pad) * w_in,
                                            *ker_loc = ker + dim_k * ker_stride[0] + dim_j * ker_stride[1] + ker_stride[1] - 1;

                                for (size_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker) {
                                    for (size_t j_ker = j_ker_start; j_ker < j_ker_end; ++j_ker) {
                                        dst_e += in_loc[i_ker * w_in + (j_dst + j_ker - w_pad)] * ker_loc[-i_ker * w_ker - j_ker];
                                    }
                                }
                            }
                            dst_loc[i_dst * w_dst + j_dst] = dst_e;
                        }
                    }
                }
            }
        }

        /** === KERNEL GRAD ===
         *
         *  in * out_grad -> kernel_grad
         *  [n, c_i, h_i, w_i] * [n, c_o, h_o, w_o] -> [c_o, c_i, h_k, w_k]
         *
         *  multithreading over: c_o
         *  vectorize: w_i (code: w_in) & w_o (code: w_ker)
         *
         *  here out_grad is referred as "ker", kernel_grad is "dst"
         */
        inline void conv_kernel_grad_fp32_worker(const size_t n, const size_t c_i, const size_t c_o,
                                                 const size_t mt_begin, const size_t mt_end,
                                                 const ssize_t h_in, const ssize_t w_in, const ssize_t h_ker, const ssize_t w_ker,
                                                 const ssize_t h_pad, const ssize_t w_pad,
                                                 float *__restrict dst, const float *__restrict in,
                                                 const float *__restrict ker) {

            const ssize_t h_dst = h_in - h_ker + 1 + h_pad * 2, w_dst = w_in - w_ker + 1 + w_pad * 2;

            const ssize_t dst_stride[2] = {static_cast<ssize_t>(c_i) * h_dst * w_dst, h_dst * w_dst},
                          in_stride[2] = {static_cast<ssize_t>(c_i) * h_in * w_in, h_in * w_in},
                          ker_stride[2] = {static_cast<ssize_t>(c_o) * h_ker * w_ker, h_ker * w_ker};

            for (size_t dim_i = mt_begin; dim_i < mt_end; ++dim_i) { // c_o
                for (size_t dim_j = 0; dim_j < c_i; ++dim_j) {
                    float *dst_loc = dst + dim_i * dst_stride[0] + dim_j * dst_stride[1]; // local offsets

                    for (ssize_t i_dst = 0; i_dst < h_dst; ++i_dst) {
                        // i_in = i_dst + i_ker - h_pad \in [0, h_in)
                        const ssize_t i_ker_start = std::max(static_cast<ssize_t>(0), h_pad - i_dst),
                                      i_ker_end = std::min(h_ker, h_in + h_pad - i_dst);

                        for (ssize_t j_dst = 0; j_dst < w_dst; ++j_dst) {
                            // j_in = j_dst + j_ker - w_pad \in [0, w_in)
                            const ssize_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                          j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            for (size_t dim_k = 0; dim_k < n; ++dim_k) {
                                // todo
                            }
                        }
                    }
                }
            }
        }


    }

    template<kernel_func::conv_mode mode>
    void conv_fp32_legacy(const size_t n, const size_t c_i, const size_t c_o,
                          const size_t h_a, const size_t w_a, const size_t h_b, const size_t w_b,
                          const size_t h_pad, const size_t w_pad,
                          char *__restrict dst_p, const char *__restrict src_a_p, const char *__restrict src_b_p,
                          const char *__restrict bias_p) noexcept {

        using enum kernel_func::conv_mode;

        if constexpr (mode == forward) {
            conv_utils_fp32::conv_fp32_worker(c_i, c_o, 0, n, h_a, w_a, h_b, w_b, h_pad, w_pad,
                                              reinterpret_cast<float *__restrict>(dst_p),
                                              reinterpret_cast<const float *__restrict>(src_a_p),
                                              reinterpret_cast<const float *__restrict>(src_b_p),
                                              reinterpret_cast<const float *__restrict>(bias_p));
            return;
        }
        else if constexpr (mode == input_grad) {
            conv_utils_fp32::conv_input_grad_fp32_worker(c_i, c_o, 0, n, h_a, w_a, h_b, w_b, h_pad, w_pad,
                                                         reinterpret_cast<float *__restrict>(dst_p),
                                                         reinterpret_cast<const float *__restrict>(src_a_p),
                                                         reinterpret_cast<const float *__restrict>(src_b_p));
            return;
        }
        else {
            // return;
        }

        const size_t h_dst = h_a - h_b + 1 + h_pad * 2, w_dst = w_a - w_b + 1 + w_pad * 2;

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
