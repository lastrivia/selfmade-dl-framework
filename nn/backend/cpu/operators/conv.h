#pragma once

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>

#include "../arch.h"
#include "../thread_pool.h"
#include "../../base_kernel.h"

#ifdef _MSC_VER
using ssize_t = ptrdiff_t;
#endif

namespace cpu_kernel {

    namespace conv_utils_fp32 {

        inline void conv_fp32_worker(
            size_t c_i, size_t c_o, size_t mt_begin, size_t mt_end,
            ssize_t h_in, ssize_t w_in, ssize_t h_ker, ssize_t w_ker, ssize_t h_pad, ssize_t w_pad,
            float *__restrict dst, const float *__restrict in, const float *__restrict ker, const float *__restrict bias
        ) noexcept;

        inline void conv_input_grad_fp32_worker(
            size_t c_i, size_t c_o, size_t mt_begin, size_t mt_end,
            ssize_t h_in, ssize_t w_in, ssize_t h_ker, ssize_t w_ker, ssize_t h_pad, ssize_t w_pad,
            float *__restrict dst, const float *__restrict in, const float *__restrict ker
        ) noexcept;

        inline void conv_kernel_grad_fp32_worker(
            size_t n, size_t c_i, size_t c_o, size_t mt_begin, size_t mt_end,
            ssize_t h_in, ssize_t w_in, ssize_t h_ker, ssize_t w_ker, ssize_t h_pad, ssize_t w_pad,
            float *__restrict dst, const float *__restrict in, const float *__restrict ker
        ) noexcept;

        inline void divide_jobs(const size_t jobs, const size_t threads, size_t *clips) { // out[threads + 1]
            size_t a = jobs / threads, b = jobs % threads;
            clips[0] = 0;
            size_t i = 1;
            for (; i <= b; ++i)
                clips[i] = clips[i - 1] + a + 1;
            for (; i <= threads; ++i)
                clips[i] = clips[i - 1] + a;
        }

        inline size_t num_of_threads(const size_t mt_dim, const size_t dims, const size_t dst_size, const size_t ker_size) {
            using namespace cpu_constants;
            static constexpr size_t CONV_WORKLOAD_FACTOR = 4;
            size_t tmp = dims * dst_size;
            if (tmp > THREAD_WORKLOAD_THRESHOLD / CONV_WORKLOAD_FACTOR * MAX_THREADS) // prevent overflow
                return std::min(mt_dim, MAX_THREADS);
            size_t threads = tmp * ker_size * CONV_WORKLOAD_FACTOR / THREAD_WORKLOAD_THRESHOLD;
            return std::min(std::max(static_cast<size_t>(1), threads), std::min(mt_dim, MAX_THREADS));
        }


        /** === FORWARD ===
         *
         *  in * kernel + bias -> out
         *  [n, c_i, h_i, w_i] * [c_o, c_i, h_k, w_k] + bias -> [n, c_o, h_o, w_o]
         *
         *  multithreading over: n
         *  vectorize: w_i (code: w_in) & w_o (code: w_dst)
         *
         *  requires padding < kernel_size
        */
        inline void conv_fp32(
            const size_t n, const size_t c_i, const size_t c_o,
            const size_t h_in, const size_t w_in, const size_t h_ker, const size_t w_ker,
            const size_t h_pad, const size_t w_pad,
            char *__restrict dst_p, const char *__restrict in_p, const char *__restrict ker_p,
            const char *__restrict bias_p
        ) noexcept {
            auto worker_call = [&](size_t mt_begin, size_t mt_end) {
                conv_fp32_worker(
                    c_i, c_o, mt_begin, mt_end,
                    static_cast<ssize_t>(h_in), static_cast<ssize_t>(w_in),
                    static_cast<ssize_t>(h_ker), static_cast<ssize_t>(w_ker),
                    static_cast<ssize_t>(h_pad), static_cast<ssize_t>(w_pad),
                    reinterpret_cast<float *__restrict>(dst_p),
                    reinterpret_cast<const float *__restrict>(in_p),
                    reinterpret_cast<const float *__restrict>(ker_p),
                    reinterpret_cast<const float *__restrict>(bias_p)
                );
            };
            const size_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;
            size_t threads = num_of_threads(n, n * c_i * c_o, h_dst * w_dst, h_ker * w_ker);
            if (threads == 1 || !ENABLE_MULTITHREADING)
                worker_call(0, n);
            else {
                std::vector<size_t> mt_clips(threads + 1);
                divide_jobs(n, threads, mt_clips.data());
                std::vector<thread_pool::task_token> tasks;
                for (size_t i = 0; i < threads; ++i)
                    tasks.push_back(thread_pool::run(
                        [&, i] { worker_call(mt_clips[i], mt_clips[i + 1]); }
                    ));
                for (auto &task: tasks)
                    task.join();
            }
        }

        inline void conv_fp32_worker(
            const size_t c_i, const size_t c_o,
            const size_t mt_begin, const size_t mt_end,
            const ssize_t h_in, const ssize_t w_in, const ssize_t h_ker, const ssize_t w_ker,
            const ssize_t h_pad, const ssize_t w_pad,
            float *__restrict dst, const float *__restrict in, const float *__restrict ker,
            const float *__restrict bias
        ) noexcept {

            const ssize_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;

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
                        for (; j_dst + static_cast<ssize_t>(AVX2_FP32_N) <= w_dst; j_dst += static_cast<ssize_t>(AVX2_FP32_N)) {

                            // j_in[] = j_dst[] + j_ker - w_pad \in [0, w_in)
                            const ssize_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst - static_cast<ssize_t>(AVX2_FP32_N) + 1),
                                          j_ker_loadu_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                          j_ker_loadu_end = std::min(w_ker, w_in + w_pad - j_dst - static_cast<ssize_t>(AVX2_FP32_N) + 1),
                                          j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            __m256 dst_vec = _mm256_set1_ps(bias[dim_j]);

                            for (size_t dim_k = 0; dim_k < c_i; ++dim_k) {
                                const float *in_loc = in + dim_i * in_stride[0] + dim_k * in_stride[1],
                                            *ker_loc = ker + dim_j * ker_stride[0] + dim_k * ker_stride[1];
                                const ptrdiff_t in_offset_r = (i_dst - h_pad) * w_in, in_offset = in_offset_r + (j_dst - w_pad);

                                for (ssize_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker) {

                                    auto boundary_load_in = [&](ssize_t j_in) {
                                        const float *in_row = in_loc + (in_offset_r + i_ker * w_in);
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
                                            _mm256_loadu_ps(in_loc + (in_offset + i_ker * w_in + j_ker)),
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

                            ssize_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                    j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            for (size_t dim_k = 0; dim_k < c_i; ++dim_k) {
                                const float *in_loc = in + dim_i * in_stride[0] + dim_k * in_stride[1],
                                            *ker_loc = ker + dim_j * ker_stride[0] + dim_k * ker_stride[1];
                                const ptrdiff_t in_offset = (i_dst - h_pad) * w_in + (j_dst - w_pad);

                                for (ssize_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker)
                                    for (ssize_t j_ker = j_ker_start; j_ker < j_ker_end; ++j_ker)
                                        dst_e += in_loc[in_offset + i_ker * w_in + j_ker] * ker_loc[i_ker * w_ker + j_ker];
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
        inline void conv_input_grad_fp32(
            const size_t n, const size_t c_i, const size_t c_o,
            const size_t h_in, const size_t w_in, const size_t h_ker, const size_t w_ker,
            const size_t h_pad, const size_t w_pad,
            char *__restrict dst_p, const char *__restrict in_p, const char *__restrict ker_p
        ) noexcept {
            auto worker_call = [&](size_t mt_begin, size_t mt_end) {
                conv_input_grad_fp32_worker(
                    c_i, c_o, mt_begin, mt_end,
                    static_cast<ssize_t>(h_in), static_cast<ssize_t>(w_in),
                    static_cast<ssize_t>(h_ker), static_cast<ssize_t>(w_ker),
                    static_cast<ssize_t>(h_pad), static_cast<ssize_t>(w_pad),
                    reinterpret_cast<float *__restrict>(dst_p),
                    reinterpret_cast<const float *__restrict>(in_p),
                    reinterpret_cast<const float *__restrict>(ker_p)
                );
            };
            const size_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;
            size_t threads = num_of_threads(n, n * c_i * c_o, h_dst * w_dst, h_ker * w_ker);
            if (threads == 1 || !ENABLE_MULTITHREADING)
                worker_call(0, n);
            else {
                std::vector<size_t> mt_clips(threads + 1);
                divide_jobs(n, threads, mt_clips.data());
                std::vector<thread_pool::task_token> tasks;
                for (size_t i = 0; i < threads; ++i)
                    tasks.push_back(thread_pool::run(
                        [&, i] { worker_call(mt_clips[i], mt_clips[i + 1]); }
                    ));
                for (auto &task: tasks)
                    task.join();
            }
        }

        inline void conv_input_grad_fp32_worker(
            const size_t c_i, const size_t c_o,
            const size_t mt_begin, const size_t mt_end,
            const ssize_t h_in, const ssize_t w_in, const ssize_t h_ker, const ssize_t w_ker,
            const ssize_t h_pad, const ssize_t w_pad,
            float *__restrict dst, const float *__restrict in, const float *__restrict ker
        ) noexcept {

            const ssize_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;

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
                        for (; j_dst + static_cast<ssize_t>(AVX2_FP32_N) <= w_dst; j_dst += static_cast<ssize_t>(AVX2_FP32_N)) {

                            // j_in[] = j_dst[] + j_ker - w_pad \in [0, w_in)
                            const ssize_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst - static_cast<ssize_t>(AVX2_FP32_N) + 1),
                                          j_ker_loadu_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                          j_ker_loadu_end = std::min(w_ker, w_in + w_pad - j_dst - static_cast<ssize_t>(AVX2_FP32_N) + 1),
                                          j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            __m256 dst_vec = _mm256_setzero_ps();

                            for (size_t dim_k = 0; dim_k < c_o; ++dim_k) {
                                const float *in_loc = in + dim_i * in_stride[0] + dim_k * in_stride[1],
                                            *ker_loc = ker + dim_k * ker_stride[0] + dim_j * ker_stride[1] + ker_stride[1] - 1;
                                const ptrdiff_t in_offset_r = (i_dst - h_pad) * w_in, in_offset = in_offset_r + (j_dst - w_pad);

                                for (ssize_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker) {

                                    auto boundary_load_in = [&](ssize_t j_in) {
                                        const float *in_row = in_loc + (in_offset_r + i_ker * w_in);
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
                                            _mm256_loadu_ps(in_loc + (in_offset + i_ker * w_in + j_ker)),
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

                            ssize_t j_ker_start = std::max(static_cast<ssize_t>(0), w_pad - j_dst),
                                    j_ker_end = std::min(w_ker, w_in + w_pad - j_dst);

                            for (size_t dim_k = 0; dim_k < c_o; ++dim_k) {
                                const float *in_loc = in + dim_i * in_stride[0] + dim_k * in_stride[1],
                                            *ker_loc = ker + dim_k * ker_stride[0] + dim_j * ker_stride[1] + ker_stride[1] - 1;
                                const ptrdiff_t in_offset = (i_dst - h_pad) * w_in + (j_dst - w_pad);

                                for (ssize_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker)
                                    for (ssize_t j_ker = j_ker_start; j_ker < j_ker_end; ++j_ker)
                                        dst_e += in_loc[in_offset + i_ker * w_in + j_ker] * ker_loc[-i_ker * w_ker - j_ker];
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
        inline void conv_kernel_grad_fp32(
            const size_t n, const size_t c_i, const size_t c_o,
            const size_t h_in, const size_t w_in, const size_t h_ker, const size_t w_ker,
            const size_t h_pad, const size_t w_pad,
            char *__restrict dst_p, const char *__restrict in_p, const char *__restrict ker_p
        ) noexcept {
            auto worker_call = [&](size_t mt_begin, size_t mt_end) {
                conv_kernel_grad_fp32_worker(
                    n, c_i, c_o, mt_begin, mt_end,
                    static_cast<ssize_t>(h_in), static_cast<ssize_t>(w_in),
                    static_cast<ssize_t>(h_ker), static_cast<ssize_t>(w_ker),
                    static_cast<ssize_t>(h_pad), static_cast<ssize_t>(w_pad),
                    reinterpret_cast<float *__restrict>(dst_p),
                    reinterpret_cast<const float *__restrict>(in_p),
                    reinterpret_cast<const float *__restrict>(ker_p)
                );
            };
            const size_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;
            size_t threads = num_of_threads(c_o, n * c_i * c_o, h_dst * w_dst, h_ker * w_ker);
            if (threads == 1 || !ENABLE_MULTITHREADING)
                worker_call(0, c_o);
            else {
                std::vector<size_t> mt_clips(threads + 1);
                divide_jobs(c_o, threads, mt_clips.data());
                std::vector<thread_pool::task_token> tasks;
                for (size_t i = 0; i < threads; ++i)
                    tasks.push_back(thread_pool::run(
                        [&, i] { worker_call(mt_clips[i], mt_clips[i + 1]); }
                    ));
                for (auto &task: tasks)
                    task.join();
            }
        }

        inline void conv_kernel_grad_fp32_worker(
            const size_t n, const size_t c_i, const size_t c_o,
            const size_t mt_begin, const size_t mt_end,
            const ssize_t h_in, const ssize_t w_in, const ssize_t h_ker, const ssize_t w_ker,
            const ssize_t h_pad, const ssize_t w_pad,
            float *__restrict dst, const float *__restrict in, const float *__restrict ker
        ) noexcept {

            const ssize_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;

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

                            float dst_e = 0.0f;

                            for (size_t dim_k = 0; dim_k < n; ++dim_k) {
                                const float *in_loc = in + dim_k * in_stride[0] + dim_j * in_stride[1],
                                            *ker_loc = ker + dim_k * ker_stride[0] + dim_i * ker_stride[1];
                                const ptrdiff_t in_offset = (i_dst - h_pad) * w_in + (j_dst - w_pad);

                                for (ssize_t i_ker = i_ker_start; i_ker < i_ker_end; ++i_ker) {
                                    ssize_t j_ker = j_ker_start;

                                    __m256 dst_vec = _mm256_setzero_ps();
                                    for (; j_ker + static_cast<ssize_t>(AVX2_FP32_N) <= j_ker_end; j_ker += static_cast<ssize_t>(AVX2_FP32_N))
                                        dst_vec = _mm256_fmadd_ps(
                                            _mm256_loadu_ps(in_loc + (in_offset + i_ker * w_in + j_ker)),
                                            _mm256_loadu_ps(ker_loc + i_ker * w_ker + j_ker),
                                            dst_vec
                                        );
                                    dst_e += horizontal_sum_avx2(dst_vec);

                                    for (; j_ker < j_ker_end; ++j_ker)
                                        dst_e += in_loc[in_offset + i_ker * w_in + j_ker] * ker_loc[i_ker * w_ker + j_ker];
                                }
                            }
                            dst_loc[i_dst * w_dst + j_dst] = dst_e;
                        }
                    }
                }
            }
        }
    }

    using conv_utils_fp32::conv_fp32;
    using conv_utils_fp32::conv_input_grad_fp32;
    using conv_utils_fp32::conv_kernel_grad_fp32;
}
