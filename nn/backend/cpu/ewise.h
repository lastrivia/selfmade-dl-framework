#pragma once

#include <cmath>

// auto simd by compiler optimization

// todo simd intrinsics

// todo cache optimization

// todo multithread

inline void cpu_kernel_add_ewise_fp32(size_t n, char *dst_p, const char *src_p_a, const char *src_p_b) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src_a = reinterpret_cast<const float *>(src_p_a);
    const auto *src_b = reinterpret_cast<const float *>(src_p_b);
    for (size_t i = 0; i < n; i++) {
        dst[i] = src_a[i] + src_b[i];
    }
}

inline void cpu_kernel_sub_ewise_fp32(size_t n, char *dst_p, const char *src_p_a, const char *src_p_b) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src_a = reinterpret_cast<const float *>(src_p_a);
    const auto *src_b = reinterpret_cast<const float *>(src_p_b);
    for (size_t i = 0; i < n; i++) {
        dst[i] = src_a[i] - src_b[i];
    }
}

inline void cpu_kernel_mul_ewise_fp32(size_t n, char *dst_p, const char *src_p_a, const char *src_p_b) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src_a = reinterpret_cast<const float *>(src_p_a);
    const auto *src_b = reinterpret_cast<const float *>(src_p_b);
    for (size_t i = 0; i < n; i++) {
        dst[i] = src_a[i] * src_b[i];
    }
}

inline void cpu_kernel_div_ewise_fp32(size_t n, char *dst_p, const char *src_p_a, const char *src_p_b) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src_a = reinterpret_cast<const float *>(src_p_a);
    const auto *src_b = reinterpret_cast<const float *>(src_p_b);
    for (size_t i = 0; i < n; i++) {
        dst[i] = src_a[i] / src_b[i];
    }
}

inline void cpu_kernel_add_scalar_fp32(size_t n, char *dst_p, const char *src_p, float scalar) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] + scalar;
    }
}

inline void cpu_kernel_mul_scalar_fp32(size_t n, char *dst_p, const char *src_p, float scalar) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] * scalar;
    }
}

inline void cpu_kernel_pow_fp32(size_t n, char *dst_p, const char *src_p, float scalar) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    for (size_t i = 0; i < n; i++) {
        dst[i] = powf(src[i], scalar);
    }
}

inline void cpu_kernel_broadcast_fp32(size_t n, char *dst_p, float val) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    for (size_t i = 0; i < n; i++) {
        dst[i] = val;
    }
}

inline void cpu_kernel_square_fp32(size_t n, char *dst_p, const char *src_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] * src[i];
    }
}

inline void cpu_kernel_sqrt_fp32(size_t n, char *dst_p, const char *src_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    for (size_t i = 0; i < n; i++) {
        dst[i] = sqrtf(src[i]);
    }
}

inline void cpu_kernel_relu_fp32(size_t n, char *dst_p, const char *src_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    for (size_t i = 0; i < n; i++) {
        dst[i] = fmaxf(src[i], 0.0f);
    }
}

inline void cpu_kernel_relu_mask_fp32(size_t n, char *dst_p, const char *src_p, const char *mask_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    const auto *mask = reinterpret_cast<const float *>(mask_p);
    for (size_t i = 0; i < n; i++) {
        dst[i] = mask[i] > 0.0f ? src[i] : 0.0f;
    }
}

inline void cpu_kernel_add_cyclic_fp32(size_t blk_n, size_t blk_len,
                                       char *dst_p, const char *src_p, const char *tile_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    const auto *tile = reinterpret_cast<const float *>(tile_p);
    for (size_t i = 0; i < blk_n; i++) {
        for (size_t j = 0; j < blk_len; j++) {
            dst[i * blk_len + j] = src[i * blk_len + j] + tile[j];
        }
    }
}

inline void cpu_kernel_sub_cyclic_fp32(size_t blk_n, size_t blk_len,
                                       char *dst_p, const char *src_p, const char *tile_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    const auto *tile = reinterpret_cast<const float *>(tile_p);
    for (size_t i = 0; i < blk_n; i++) {
        for (size_t j = 0; j < blk_len; j++) {
            dst[i * blk_len + j] = src[i * blk_len + j] - tile[j];
        }
    }
}

inline void cpu_kernel_add_stretched_fp32(size_t blk_n, size_t blk_len,
                                       char *dst_p, const char *src_p, const char *tile_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    const auto *tile = reinterpret_cast<const float *>(tile_p);
    for (size_t i = 0; i < blk_n; i++) {
        for (size_t j = 0; j < blk_len; j++) {
            dst[i * blk_len + j] = src[i * blk_len + j] + tile[i];
        }
    }
}

inline void cpu_kernel_sub_stretched_fp32(size_t blk_n, size_t blk_len,
                                       char *dst_p, const char *src_p, const char *tile_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    const auto *tile = reinterpret_cast<const float *>(tile_p);
    for (size_t i = 0; i < blk_n; i++) {
        for (size_t j = 0; j < blk_len; j++) {
            dst[i * blk_len + j] = src[i * blk_len + j] - tile[i];
        }
    }
}

inline void cpu_kernel_sum_cyclic_fp32(size_t blk_n, size_t blk_len, char *dst_p, const char *src_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    memset(dst, 0, blk_len * sizeof(float));
    for (size_t i = 0; i < blk_n; i++) {
        for (size_t j = 0; j < blk_len; j++) {
            dst[j] += src[i * blk_len + j];
        }
    }
}

inline void cpu_kernel_sum_stretched_fp32(size_t blk_n, size_t blk_len, char* dst_p, const char* src_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    memset(dst, 0, blk_n * sizeof(float));
    for (size_t i = 0; i < blk_n; i++) {
        for (size_t j = 0; j < blk_len; j++) {
            dst[i] += src[i * blk_len + j];
        }
    }
}

inline void cpu_kernel_softmax_fp32(size_t blk_n, size_t blk_len, char* dst_p, const char* src_p) noexcept {
    auto *dst = reinterpret_cast<float *>(dst_p);
    const auto *src = reinterpret_cast<const float *>(src_p);
    for (size_t i = 0; i < blk_n; i++) {
        float* row_dst = dst + i * blk_len;
        const float* row_src = src + i * blk_len;

        float max_val = row_src[0];
        for (size_t j = 0; j < blk_len; j++)
            max_val = std::max(max_val, row_src[j]);

        float sum = 0.0f;
        for (size_t j = 0; j < blk_len; j++) {
            row_dst[j] = expf(row_src[j] - max_val);
            sum += row_dst[j];
        }

        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < blk_len; j++)
            row_dst[j] *= inv_sum;
    }
}
