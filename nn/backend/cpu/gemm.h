#pragma once
#include <cstring>

// todo matrix partition & strassen algorithm

// todo cache optimization

// todo multithread

template<bool transpose_a, bool transpose_b> // only affect column/row-major
void cpu_kernel_gemm_fp32(size_t m, size_t k, size_t n, char *dst_p, const char *src_p_a, const char *src_p_b) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p); // m * n
    const auto *src_a = reinterpret_cast<const float *>(src_p_a); // m * k
    const auto *src_b = reinterpret_cast<const float *>(src_p_b); // k * n
    memset(dst, 0, n * m * sizeof(float));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float *dst_e = dst + i * n + j;
            for (size_t t = 0; t < k; ++t) {
                // dst(i, j) += a(i, t) * b(t, j)
                *dst_e +=
                        src_a[transpose_a ? t * m + i : i * k + t] *
                        src_b[transpose_b ? j * k + t : t * n + j];
            }
        }
    }
}
