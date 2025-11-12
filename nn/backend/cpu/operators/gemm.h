#pragma once

#include <immintrin.h>
#include <cstring>

#include "../arch.h"
#include "../thread_pool.h"
#include "../../mem_pool.h"


namespace cpu_kernel {

    namespace gemm_utils_fp32 {

        // stride: offset between adjacent rows(row-major) or cols(col-major)

        // transpose tag: only affects if data is (false:)row-major or (true:)col-major

        template<bool tr_a, bool tr_b>
        void plain_matmul(const size_t m, const size_t p, const size_t n,
                          float *__restrict dst, const float *__restrict src_a, const float *__restrict src_b,
                          const size_t dst_stride, const size_t a_stride, const size_t b_stride) {

            if constexpr (!tr_b) {
                // a: row/col, b: row
                // vectorize n(j) dimension

                for (size_t i = 0; i < m; ++i) {
                    size_t j = 0;

                    for (; j + AVX2_FP32_N <= n; j += AVX2_FP32_N) {
                        __m256 dst_vec = _mm256_setzero_ps();
                        for (size_t k = 0; k < p; ++k) {
                            __m256 a_vec = _mm256_set1_ps(tr_a ? src_a[k * a_stride + i] : src_a[i * a_stride + k]);
                            __m256 b_vec = _mm256_loadu_ps(src_b + k * b_stride + j);
                            dst_vec = _mm256_fmadd_ps(a_vec, b_vec, dst_vec);
                        }
                        _mm256_storeu_ps(dst + i * dst_stride + j, dst_vec);
                    }

                    if (j < n) {
                        size_t remaining = n - j;
                        __m256 dst_vec = _mm256_setzero_ps();
                        for (size_t k = 0; k < p; ++k) {
                            __m256 a_vec = _mm256_set1_ps(tr_a ? src_a[k * a_stride + i] : src_a[i * a_stride + k]);
                            alignas(32) float b_tmp[AVX2_FP32_N];
                            for (size_t s = 0; s < remaining; ++s)
                                b_tmp[s] = src_b[k * b_stride + j + s];
                            __m256 b_vec = _mm256_load_ps(b_tmp);
                            dst_vec = _mm256_fmadd_ps(a_vec, b_vec, dst_vec);
                        }
                        alignas(32) float dst_tmp[AVX2_FP32_N];
                        _mm256_store_ps(dst_tmp, dst_vec);
                        for (size_t s = 0; s < remaining; ++s)
                            dst[i * dst_stride + j + s] = dst_tmp[s];
                    }
                }
            }
            else if constexpr (!tr_a) {
                // a: row, b: col
                // vectorize p(k) dimension

                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        size_t k = 0;
                        __m256 dst_vec = _mm256_setzero_ps();

                        for (; k + AVX2_FP32_N <= p; k += AVX2_FP32_N) {
                            __m256 a_vec = _mm256_loadu_ps(src_a + i * a_stride + k);
                            __m256 b_vec = _mm256_loadu_ps(src_b + j * b_stride + k);
                            dst_vec = _mm256_fmadd_ps(a_vec, b_vec, dst_vec);
                        }
                        float dst_f = horizontal_sum_avx2(dst_vec);

                        for (; k < p; ++k)
                            dst_f += src_a[i * a_stride + k] * src_b[j * b_stride + k];
                        dst[i * dst_stride + j] = dst_f;
                    }
                }
            }
            else {
                // a: col, b: col
                // vectorize m(i) dimension

                for (size_t j = 0; j < n; ++j) {
                    size_t i = 0;

                    for (; i + AVX2_FP32_N <= m; i += AVX2_FP32_N) {
                        __m256 dst_vec = _mm256_setzero_ps();
                        for (size_t k = 0; k < p; ++k) {
                            __m256 b_vec = _mm256_set1_ps(src_b[j * b_stride + k]);
                            __m256 a_vec = _mm256_loadu_ps(src_a + k * a_stride + i);
                            dst_vec = _mm256_fmadd_ps(a_vec, b_vec, dst_vec);
                        }
                        _mm256_storeu_ps(dst + i * dst_stride + j, dst_vec);
                    }

                    if (i < m) {
                        size_t remaining = m - i;
                        __m256 dst_vec = _mm256_setzero_ps();
                        for (size_t k = 0; k < p; ++k) {
                            __m256 b_vec = _mm256_set1_ps(src_b[j * b_stride + k]);
                            alignas(32) float a_tmp[AVX2_FP32_N];
                            for (size_t s = 0; s < remaining; ++s)
                                a_tmp[s] = src_a[k * a_stride + i + s];
                            __m256 a_vec = _mm256_load_ps(a_tmp);
                            dst_vec = _mm256_fmadd_ps(a_vec, b_vec, dst_vec);
                        }
                        alignas(32) float dst_tmp[AVX2_FP32_N];
                        _mm256_store_ps(dst_tmp, dst_vec);
                        for (size_t s = 0; s < remaining; ++s)
                            dst[i * dst_stride + j + s] = dst_tmp[s];
                    }
                }
            }
        }

        inline void plain_add(const size_t blk_n, const size_t blk_len,
                              float *__restrict dst, const float *__restrict src_a, const float *__restrict src_b,
                              const size_t dst_stride, const size_t src_stride) {
            for (size_t i = 0; i < blk_n; ++i) {
                size_t src_offset = i * src_stride, dst_offset = i * dst_stride;
                size_t j = 0;
                for (; j + AVX2_FP32_N <= blk_len; j += AVX2_FP32_N) {
                    __m256 a_vec = _mm256_loadu_ps(src_a + src_offset + j);
                    __m256 b_vec = _mm256_loadu_ps(src_b + src_offset + j);
                    _mm256_storeu_ps(dst + dst_offset + j, _mm256_add_ps(a_vec, b_vec));
                }
                for (; j < blk_len; ++j) {
                    dst[dst_offset + j] = src_a[src_offset + j] + src_b[src_offset + j];
                }
            }
        }

        inline void plain_sub(const size_t blk_n, const size_t blk_len,
                              float *__restrict dst, const float *__restrict src_a, const float *__restrict src_b,
                              const size_t dst_stride, const size_t src_stride) {
            for (size_t i = 0; i < blk_n; ++i) {
                size_t src_offset = i * src_stride, dst_offset = i * dst_stride;
                size_t j = 0;
                for (; j + AVX2_FP32_N <= blk_len; j += AVX2_FP32_N) {
                    __m256 a_vec = _mm256_loadu_ps(src_a + src_offset + j);
                    __m256 b_vec = _mm256_loadu_ps(src_b + src_offset + j);
                    _mm256_storeu_ps(dst + dst_offset + j, _mm256_sub_ps(a_vec, b_vec));
                }
                for (; j < blk_len; ++j) {
                    dst[dst_offset + j] = src_a[src_offset + j] - src_b[src_offset + j];
                }
            }
        }

        inline void plain_add_add_sub(const size_t blk_n, const size_t blk_len,
                                      float *__restrict dst, const float *__restrict src_a, const float *__restrict src_b,
                                      const float *__restrict src_c, const float *__restrict src_sub,
                                      const size_t dst_stride, const size_t src_stride) {
            for (size_t i = 0; i < blk_n; ++i) {
                size_t src_offset = i * src_stride, dst_offset = i * dst_stride;
                size_t j = 0;
                for (; j + AVX2_FP32_N <= blk_len; j += AVX2_FP32_N) {
                    __m256 a_vec = _mm256_loadu_ps(src_a + src_offset + j);
                    __m256 b_vec = _mm256_loadu_ps(src_b + src_offset + j);
                    __m256 c_vec = _mm256_loadu_ps(src_c + src_offset + j);
                    __m256 sub_vec = _mm256_loadu_ps(src_sub + src_offset + j);
                    _mm256_storeu_ps(
                        dst + dst_offset + j,
                        _mm256_add_ps(_mm256_add_ps(a_vec, b_vec), _mm256_sub_ps(c_vec, sub_vec))
                    );
                }
                for (; j < blk_len; ++j) {
                    dst[dst_offset + j] = src_a[src_offset + j] + src_b[src_offset + j] + src_c[src_offset + j] - src_sub[src_offset + j];
                }
            }
        }

        template<bool tr_a, bool tr_b>
        void partition(const size_t m, const size_t p, const size_t n,
                       float *dst, const float *src_a, const float *src_b,
                       const size_t dst_stride, const size_t a_stride, const size_t b_stride) {

            if ((m & 1) || (p & 1) || (n & 1) ||
                (m * n * p < THREAD_WORKLOAD_THRESHOLD * 7 && (m * n + n * p + p * m) < CACHE_THRESHOLD / sizeof(float))) {

                plain_matmul<tr_a, tr_b>(m, p, n, dst, src_a, src_b, dst_stride, a_stride, b_stride);
                return;
            }

            auto blk_a = [m, p, src_a, a_stride](const int i, const int j) -> const float * {
                return src_a + (tr_a ? i * (m / 2) + j * (p / 2) * a_stride : i * (m / 2) * a_stride + j * (p / 2));
            };
            auto blk_b = [p, n, src_b, b_stride](const int i, const int j) -> const float * {
                return src_b + (tr_b ? i * (p / 2) + j * (n / 2) * b_stride : i * (p / 2) * b_stride + j * (n / 2));
            };
            auto blk_dst = [m, n, dst, dst_stride](const int i, const int j) -> float * {
                return dst + i * (m / 2) * dst_stride + j * (n / 2);
            };

            float *tmp_a[5], *tmp_b[5], *tmp_m[7];
            for (auto &i: tmp_a)
                i = mem_pool::alloc<float>(m * p / 4);
            for (auto &i: tmp_b)
                i = mem_pool::alloc<float>(p * n / 4);
            for (auto &i: tmp_m)
                i = mem_pool::alloc<float>(m * n / 4);

            const size_t tmp_a_stride = tr_a ? m / 2 : p / 2, tmp_b_stride = tr_b ? p / 2 : n / 2;

            plain_add(tr_a ? p / 2 : m / 2, tmp_a_stride, tmp_a[0], blk_a(0, 0), blk_a(1, 1), tmp_a_stride, a_stride);
            plain_add(tr_a ? p / 2 : m / 2, tmp_a_stride, tmp_a[1], blk_a(1, 0), blk_a(1, 1), tmp_a_stride, a_stride);
            plain_add(tr_a ? p / 2 : m / 2, tmp_a_stride, tmp_a[2], blk_a(0, 0), blk_a(0, 1), tmp_a_stride, a_stride);
            plain_sub(tr_a ? p / 2 : m / 2, tmp_a_stride, tmp_a[3], blk_a(1, 0), blk_a(0, 0), tmp_a_stride, a_stride);
            plain_sub(tr_a ? p / 2 : m / 2, tmp_a_stride, tmp_a[4], blk_a(0, 1), blk_a(1, 1), tmp_a_stride, a_stride);

            plain_add(tr_b ? n / 2 : p / 2, tmp_b_stride, tmp_b[0], blk_b(0, 0), blk_b(1, 1), tmp_b_stride, b_stride);
            plain_sub(tr_b ? n / 2 : p / 2, tmp_b_stride, tmp_b[1], blk_b(0, 1), blk_b(1, 1), tmp_b_stride, b_stride);
            plain_sub(tr_b ? n / 2 : p / 2, tmp_b_stride, tmp_b[2], blk_b(1, 0), blk_b(0, 0), tmp_b_stride, b_stride);
            plain_add(tr_b ? n / 2 : p / 2, tmp_b_stride, tmp_b[3], blk_b(0, 0), blk_b(0, 1), tmp_b_stride, b_stride);
            plain_add(tr_b ? n / 2 : p / 2, tmp_b_stride, tmp_b[4], blk_b(1, 0), blk_b(1, 1), tmp_b_stride, b_stride);

            std::vector<std::function<void()> > calls = {
                std::bind(partition<tr_a, tr_b>, m / 2, p / 2, n / 2, tmp_m[0], tmp_a[0], tmp_b[0], n / 2, tmp_a_stride, tmp_b_stride),
                std::bind(partition<tr_a, tr_b>, m / 2, p / 2, n / 2, tmp_m[1], tmp_a[1], blk_b(0, 0), n / 2, tmp_a_stride, b_stride),
                std::bind(partition<tr_a, tr_b>, m / 2, p / 2, n / 2, tmp_m[2], blk_a(0, 0), tmp_b[1], n / 2, a_stride, tmp_b_stride),
                std::bind(partition<tr_a, tr_b>, m / 2, p / 2, n / 2, tmp_m[3], blk_a(1, 1), tmp_b[2], n / 2, a_stride, tmp_b_stride),
                std::bind(partition<tr_a, tr_b>, m / 2, p / 2, n / 2, tmp_m[4], tmp_a[2], blk_b(1, 1), n / 2, tmp_a_stride, b_stride),
                std::bind(partition<tr_a, tr_b>, m / 2, p / 2, n / 2, tmp_m[5], tmp_a[3], tmp_b[3], n / 2, tmp_a_stride, tmp_b_stride),
                std::bind(partition<tr_a, tr_b>, m / 2, p / 2, n / 2, tmp_m[6], tmp_a[4], tmp_b[4], n / 2, tmp_a_stride, tmp_b_stride)
            };

            if constexpr (ENABLE_MULTITHREADING) {
                std::vector<thread_pool::task_token> tasks;
                for (auto &call: calls)
                    tasks.push_back(thread_pool::run(call));
                for (auto &task: tasks)
                    task.join();
            }
            else {
                for (auto &call: calls)
                    call();
            }

            plain_add_add_sub(m / 2, n / 2, blk_dst(0, 0), tmp_m[0], tmp_m[3], tmp_m[6], tmp_m[4], n, n / 2);
            plain_add(m / 2, n / 2, blk_dst(0, 1), tmp_m[2], tmp_m[4], n, n / 2);
            plain_add(m / 2, n / 2, blk_dst(1, 0), tmp_m[1], tmp_m[3], n, n / 2);
            plain_add_add_sub(m / 2, n / 2, blk_dst(1, 1), tmp_m[0], tmp_m[2], tmp_m[5], tmp_m[1], n, n / 2);

            for (auto &i: tmp_a)
                mem_pool::recycle(i);
            for (auto &i: tmp_b)
                mem_pool::recycle(i);
            for (auto &i: tmp_m)
                mem_pool::recycle(i);
        }
    }

    template<bool transpose_a, bool transpose_b>
    void gemm_fp32(size_t m, size_t p, size_t n, float *dst, const float *src_a, const float *src_b) {

        gemm_utils_fp32::partition<transpose_a, transpose_b>(
            m, p, n, dst, src_a, src_b, n, transpose_a ? m : p, transpose_b ? p : n
        );
    }
}
