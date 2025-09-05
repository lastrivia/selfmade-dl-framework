#pragma once

#include <immintrin.h>
#include <cstring>

#include "constants.h"
#include "thread_pool.h"

class cpu_kernel_gemm_fp32_utils {
public:
    template<bool, bool>
    friend void cpu_kernel_gemm_fp32(size_t, size_t, size_t, char *, const char *, const char *) noexcept;

    cpu_kernel_gemm_fp32_utils() = delete;

private:
    // stride: offset between adjacent rows(row-major) or cols(col-major)

    // transpose tag: only affects if data is (false:)row-major or (true:)col-major

    template<bool tr_a, bool tr_b>
    static void plain_matmul(const size_t m, const size_t p, const size_t n,
                             float *dst, const float *src_a, const float *src_b,
                             const size_t dst_stride, const size_t a_stride, const size_t b_stride) noexcept {

        // for (size_t i = 0; i < m; ++i) {
        //     for (size_t j = 0; j < n; ++j) {
        //         float *dst_e = dst + i * dst_stride + j;
        //         *dst_e = 0;
        //         for (size_t k = 0; k < p; ++k) {
        //             // dst(i, j) += a(i, k) * b(k, j)
        //             *dst_e +=
        //                     src_a[tr_a ? k * a_stride + i : i * a_stride + k] *
        //                     src_b[tr_b ? j * b_stride + k : k * b_stride + j];
        //         }
        //     }
        // }

        static constexpr size_t AVX2_FP32_N = 8;

        // todo optimize transposed_b cases
        for (size_t i = 0; i < m; ++i) {
            size_t j = 0;

            for (; j + AVX2_FP32_N <= n; j += AVX2_FP32_N) {
                __m256 dst_vec = _mm256_setzero_ps();

                for (size_t k = 0; k < p; ++k) {
                    float a_val = tr_a ? src_a[k * a_stride + i] : src_a[i * a_stride + k];
                    __m256 a_vec = _mm256_set1_ps(a_val);

                    __m256 b_vec;
                    if constexpr (!tr_b) {
                        b_vec = _mm256_loadu_ps(src_b + k * b_stride + j);
                    }
                    else {
                        alignas(32) float tmp[AVX2_FP32_N];
                        for (size_t s = 0; s < AVX2_FP32_N; ++s)
                            tmp[s] = src_b[(j + s) * b_stride + k];
                        b_vec = _mm256_load_ps(tmp);
                    }

                    dst_vec = _mm256_fmadd_ps(a_vec, b_vec, dst_vec);
                }

                _mm256_storeu_ps(dst + i * dst_stride + j, dst_vec);
            }

            if (j < n) {
                size_t remaining = n - j;
                alignas(32) float dst_tmp[AVX2_FP32_N] = {0};

                for (size_t t = 0; t < p; ++t) {
                    float a_val = tr_a ? src_a[t * a_stride + i] : src_a[i * a_stride + t];
                    __m256 a_vec = _mm256_set1_ps(a_val);

                    alignas(32) float b_tmp[AVX2_FP32_N] = {0};
                    for (size_t s = 0; s < remaining; ++s) {
                        if constexpr (!tr_b)
                            b_tmp[s] = src_b[t * b_stride + j + s];
                        else
                            b_tmp[s] = src_b[(j + s) * b_stride + t];
                    }
                    __m256 b_vec = _mm256_load_ps(b_tmp);

                    __m256 dst_vec = _mm256_load_ps(dst_tmp);
                    dst_vec = _mm256_fmadd_ps(a_vec, b_vec, dst_vec);
                    _mm256_store_ps(dst_tmp, dst_vec);
                }

                for (size_t s = 0; s < remaining; ++s)
                    dst[i * dst_stride + j + s] = dst_tmp[s];
            }
        }
    }

    // todo simd
    static void plain_add(const size_t blk_n, const size_t blk_len,
                          float *dst, const float *src_a, const float *src_b,
                          const size_t dst_stride, const size_t src_stride) noexcept {
        for (int i = 0; i < blk_n; ++i) {
            for (int j = 0; j < blk_len; ++j) {
                dst[i * dst_stride + j] = src_a[i * src_stride + j] + src_b[i * src_stride + j];
            }
        }
    }

    static void plain_sub(const size_t blk_n, const size_t blk_len,
                          float *dst, const float *src_a, const float *src_b,
                          const size_t dst_stride, const size_t src_stride) noexcept {
        for (int i = 0; i < blk_n; ++i) {
            for (int j = 0; j < blk_len; ++j) {
                dst[i * dst_stride + j] = src_a[i * src_stride + j] - src_b[i * src_stride + j];
            }
        }
    }

    static void plain_add_add_sub(const size_t blk_n, const size_t blk_len,
                                  float *dst, const float *src_a, const float *src_b, const float *src_c, const float *src_sub,
                                  const size_t dst_stride, const size_t src_stride) noexcept {
        for (int i = 0; i < blk_n; ++i) {
            for (int j = 0; j < blk_len; ++j) {
                dst[i * dst_stride + j] = src_a[i * src_stride + j] + src_b[i * src_stride + j] + src_c[i * src_stride + j] - src_sub[i * src_stride + j];
            }
        }
    }

    template<bool tr_a, bool tr_b, bool can_issue_threads>
    static void partition(const size_t m, const size_t p, const size_t n,
                          float *dst, const float *src_a, const float *src_b,
                          const size_t dst_stride, const size_t a_stride, const size_t b_stride) noexcept {

        if ((m & 1) || (p & 1) || (n & 1) ||
            (m * n * p < cpu_constants::THREAD_FLOPS_THRESHOLD * 7 && (m * n + n * p + p * m) < cpu_constants::CACHE_THRESHOLD / sizeof(float))) {

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
            i = new float[m * p / 4];
        for (auto &i: tmp_b)
            i = new float[p * n / 4];
        for (auto &i: tmp_m)
            i = new float[m * n / 4];

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
            std::bind(partition<tr_a, tr_b, false>, m / 2, p / 2, n / 2, tmp_m[0], tmp_a[0], tmp_b[0], n / 2, tmp_a_stride, tmp_b_stride),
            std::bind(partition<tr_a, tr_b, false>, m / 2, p / 2, n / 2, tmp_m[1], tmp_a[1], blk_b(0, 0), n / 2, tmp_a_stride, b_stride),
            std::bind(partition<tr_a, tr_b, false>, m / 2, p / 2, n / 2, tmp_m[2], blk_a(0, 0), tmp_b[1], n / 2, a_stride, tmp_b_stride),
            std::bind(partition<tr_a, tr_b, false>, m / 2, p / 2, n / 2, tmp_m[3], blk_a(1, 1), tmp_b[2], n / 2, a_stride, tmp_b_stride),
            std::bind(partition<tr_a, tr_b, false>, m / 2, p / 2, n / 2, tmp_m[4], tmp_a[2], blk_b(1, 1), n / 2, tmp_a_stride, b_stride),
            std::bind(partition<tr_a, tr_b, false>, m / 2, p / 2, n / 2, tmp_m[5], tmp_a[3], tmp_b[3], n / 2, tmp_a_stride, tmp_b_stride),
            std::bind(partition<tr_a, tr_b, false>, m / 2, p / 2, n / 2, tmp_m[6], tmp_a[4], tmp_b[4], n / 2, tmp_a_stride, tmp_b_stride)
        };

        if constexpr (can_issue_threads) {
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
            delete[] i;
        for (auto &i: tmp_b)
            delete[] i;
        for (auto &i: tmp_m)
            delete[] i;
    }
};


template<bool transpose_a, bool transpose_b>
void cpu_kernel_gemm_fp32(size_t m, size_t p, size_t n, char *dst_p, const char *src_p_a, const char *src_p_b) noexcept {

    cpu_kernel_gemm_fp32_utils::partition<transpose_a, transpose_b, true>(
        m, p, n, reinterpret_cast<float *>(dst_p), reinterpret_cast<const float *>(src_p_a), reinterpret_cast<const float *>(src_p_b),
        n, transpose_a ? m : p, transpose_b ? p : n
    );
}
