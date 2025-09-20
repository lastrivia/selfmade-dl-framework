#pragma once

#include <cmath>

// auto simd by compiler optimization

// todo simd intrinsics, cache optimization, multithreading

namespace cpu_kernel {

    inline void add_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src_a[i] + src_b[i];
        }
    }

    inline void sub_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src_a[i] - src_b[i];
        }
    }

    inline void mul_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src_a[i] * src_b[i];
        }
    }

    inline void div_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src_a[i] / src_b[i];
        }
    }

    inline void add_scalar_fp32(size_t n, float *dst, const float *src, float scalar) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src[i] + scalar;
        }
    }

    inline void mul_scalar_fp32(size_t n, float *dst, const float *src, float scalar) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src[i] * scalar;
        }
    }

    inline void pow_fp32(size_t n, float *dst, const float *src, float scalar) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = powf(src[i], scalar);
        }
    }

    inline void broadcast_fp32(size_t n, float *dst, float val) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = val;
        }
    }

    inline void square_fp32(size_t n, float *dst, const float *src) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src[i] * src[i];
        }
    }

    inline void sqrt_fp32(size_t n, float *dst, const float *src) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = sqrtf(src[i]);
        }
    }

    inline void relu_fp32(size_t n, float *dst, const float *src) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = fmaxf(src[i], 0.0f);
        }
    }

    inline void relu_backward_fp32(size_t n, float *dst, const float *src, const float *mask) noexcept {
        for (size_t i = 0; i < n; i++) {
            dst[i] = mask[i] > 0.0f ? src[i] : 0.0f;
        }
    }

    inline void add_cyclic_fp32(size_t blk_n, size_t blk_len,
                                float *dst, const float *src, const float *tile) noexcept {
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i * blk_len + j] = src[i * blk_len + j] + tile[j];
            }
        }
    }

    inline void sub_cyclic_fp32(size_t blk_n, size_t blk_len,
                                float *dst, const float *src, const float *tile) noexcept {
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i * blk_len + j] = src[i * blk_len + j] - tile[j];
            }
        }
    }

    inline void add_stretched_fp32(size_t blk_n, size_t blk_len,
                                   float *dst, const float *src, const float *tile) noexcept {
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i * blk_len + j] = src[i * blk_len + j] + tile[i];
            }
        }
    }

    inline void sub_stretched_fp32(size_t blk_n, size_t blk_len,
                                   float *dst, const float *src, const float *tile) noexcept {
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i * blk_len + j] = src[i * blk_len + j] - tile[i];
            }
        }
    }

    inline void sum_cyclic_fp32(size_t blk_n, size_t blk_len, float *dst, const float *src) noexcept {
        memset(dst, 0, blk_len * sizeof(float));
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[j] += src[i * blk_len + j];
            }
        }
    }

    inline void sum_stretched_fp32(size_t blk_n, size_t blk_len, float *dst, const float *src) noexcept {
        memset(dst, 0, blk_n * sizeof(float));
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i] += src[i * blk_len + j];
            }
        }
    }

    inline void softmax_fp32(size_t blk_n, size_t blk_len, float *dst, const float *src) noexcept {
        for (size_t i = 0; i < blk_n; i++) {
            float *row_dst = dst + i * blk_len;
            const float *row_src = src + i * blk_len;

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

    inline void maxpool_fp32(size_t channel_n, size_t h, size_t w, size_t h_stride, size_t w_stride,
                             float *dst, bool *mask, const float *src) noexcept {
        size_t h_out = (h - 1) / h_stride + 1;
        size_t w_out = (w - 1) / w_stride + 1;

        memset(dst, 0, channel_n * h_out * w_out * sizeof(float));
        memset(mask, 0, channel_n * h * w * sizeof(bool));

        for (size_t i = 0; i < channel_n; i++) {
            for (size_t j = 0; j < h_out; j++) {
                for (size_t k = 0; k < w_out; k++) {

                    size_t local_offset = (i * h + j * h_stride) * w + k * w_stride;
                    size_t h_local = std::min(h_stride, h - j * h_stride);
                    size_t w_local = std::min(w_stride, w - k * w_stride);

                    float max_val = src[local_offset];
                    size_t max_h = 0, max_w = 0;

                    for (size_t j_local = 0; j_local < h_local; j_local++) {
                        for (size_t k_local = 0; k_local < w_local; k_local++) {
                            if (src[local_offset + j_local * w + k_local] > max_val) {
                                max_val = src[local_offset + j_local * w + k_local];
                                max_h = j_local;
                                max_w = k_local;
                            }
                        }
                    }

                    mask[local_offset + max_h * w + max_w] = true;
                    dst[(i * h_out + j) * w_out + k] = max_val;
                }
            }
        }
    }

    inline void maxpool_backward_fp32(size_t channel_n, size_t h, size_t w, size_t h_stride, size_t w_stride,
                                      float *dst, const bool *mask, const float *src) noexcept {
        auto *dst_bits = reinterpret_cast<int32_t *>(dst);
        const auto *src_bits = reinterpret_cast<const int32_t *>(src);

        size_t h_src = (h - 1) / h_stride + 1;
        size_t w_src = (w - 1) / w_stride + 1;

        for (size_t i = 0; i < channel_n; i++) {
            for (size_t j = 0; j < h_src; j++) {
                for (size_t k = 0; k < w_src; k++) {

                    size_t src_offset = (i * h_src + j) * w_src + k;
                    size_t dst_offset = (i * h + j * h_stride) * w + k * w_stride;

                    size_t h_local = std::min(h_stride, h - j * h_stride);
                    size_t w_local = std::min(w_stride, w - k * w_stride);

                    for (size_t j_local = 0; j_local < h_local; j_local++) {
                        for (size_t k_local = 0; k_local < w_local; k_local++) {
                            // bitmask trick:
                            // dst[dst_offset + j_local * w + k_local] =
                            //     mask[dst_offset + j_local * w + k_local] ? src[src_offset] : 0.0f
                            int32_t bitmask = -static_cast<int32_t>(mask[dst_offset + j_local * w + k_local]);
                            dst_bits[dst_offset + j_local * w + k_local] = src_bits[src_offset] & bitmask;
                        }
                    }
                }
            }
        }
    }
}
