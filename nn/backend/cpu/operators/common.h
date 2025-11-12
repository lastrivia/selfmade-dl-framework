#pragma once

#include "../../mem_pool.h"
#include "../arch.h"

// auto simd by compiler optimization

// todo simd intrinsics, cache optimization, multithreading

namespace cpu_kernel {

    template<typename T>
        requires std::is_trivially_copyable_v<T>
    void copy_raw(size_t n, T *dst, const T *src) {
        memcpy(dst, src, n * sizeof(T));
    }

    template<typename T>
    void add_ewise(size_t n, T *dst, const T *src_a, const T *src_b) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src_a[i] + src_b[i];
        }
    }

    inline void sub_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src_a[i] - src_b[i];
        }
    }

    inline void mul_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src_a[i] * src_b[i];
        }
    }

    inline void div_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src_a[i] / src_b[i];
        }
    }

    inline void add_scalar_fp32(size_t n, float *dst, const float *src, float scalar) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src[i] + scalar;
        }
    }

    inline void mul_scalar_fp32(size_t n, float *dst, const float *src, float scalar) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src[i] * scalar;
        }
    }

    inline void pow_fp32(size_t n, float *dst, const float *src, float scalar) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = powf(src[i], scalar);
        }
    }

    inline void broadcast_fp32(size_t n, float *dst, float val) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = val;
        }
    }

    inline void broadcast_int32(size_t n, int32_t *dst, int32_t val) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = val;
        }
    }

    inline void square_fp32(size_t n, float *dst, const float *src) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = src[i] * src[i];
        }
    }

    inline void sqrt_fp32(size_t n, float *dst, const float *src) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = sqrtf(src[i]);
        }
    }

    inline void relu_fp32(size_t n, float *dst, const float *src) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = fmaxf(src[i], 0.0f);
        }
    }

    inline void relu_backward_fp32(size_t n, float *dst, const float *src, const float *mask) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = mask[i] > 0.0f ? src[i] : 0.0f;
        }
    }

    inline void add_cyclic_fp32(size_t blk_n, size_t blk_len,
                                float *dst, const float *src, const float *tile) {
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i * blk_len + j] = src[i * blk_len + j] + tile[j];
            }
        }
    }

    inline void sub_cyclic_fp32(size_t blk_n, size_t blk_len,
                                float *dst, const float *src, const float *tile) {
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i * blk_len + j] = src[i * blk_len + j] - tile[j];
            }
        }
    }

    inline void add_stretched_fp32(size_t blk_n, size_t blk_len,
                                   float *dst, const float *src, const float *tile) {
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i * blk_len + j] = src[i * blk_len + j] + tile[i];
            }
        }
    }

    inline void sub_stretched_fp32(size_t blk_n, size_t blk_len,
                                   float *dst, const float *src, const float *tile) {
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i * blk_len + j] = src[i * blk_len + j] - tile[i];
            }
        }
    }

    inline void sum_cyclic_fp32(size_t blk_n, size_t blk_len, float *dst, const float *src) {
        memset(dst, 0, blk_len * sizeof(float));
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[j] += src[i * blk_len + j];
            }
        }
    }

    inline void sum_stretched_fp32(size_t blk_n, size_t blk_len, float *dst, const float *src) {
        memset(dst, 0, blk_n * sizeof(float));
        for (size_t i = 0; i < blk_n; i++) {
            for (size_t j = 0; j < blk_len; j++) {
                dst[i] += src[i * blk_len + j];
            }
        }
    }

    inline void add_broadcast_fp32(size_t n, size_t ndim, const size_t *lengths, const bool *mask_a, const bool *mask_b,
                                   float *dst, const float *src_a, const float *src_b) {
        size_t ndim_buf[3][NDIM_STACK_BUF_SIZE];

        workspace ndim_workspace(device_type::cpu);
        size_t *strides_a, *strides_b, *coord;
        if (ndim > NDIM_STACK_BUF_SIZE) {
            ndim_workspace.init(sizeof(size_t) * ndim * 3);
            strides_a = static_cast<size_t *>(ndim_workspace);
            strides_b = static_cast<size_t *>(ndim_workspace) + ndim;
            coord = static_cast<size_t *>(ndim_workspace) + ndim * 2;
        }
        else {
            strides_a = ndim_buf[0];
            strides_b = ndim_buf[1];
            coord = ndim_buf[2];
        }
        calc_strides(strides_a, ndim, lengths, mask_a);
        calc_strides(strides_b, ndim, lengths, mask_b);

        memset(coord, 0, ndim * sizeof(size_t));
        size_t idx_a = 0, idx_b = 0;
        for (size_t i = 0; i < n; i++) {
            // todo simd
            dst[i] = src_a[idx_a] + src_b[idx_b];
            for (size_t j = 0; j < ndim; ++j) {
                coord[j]++;
                if (coord[j] < lengths[j]) {
                    idx_a += strides_a[j];
                    idx_b += strides_b[j];
                    break;
                }
                // else: coordination carry
                coord[j] = 0;
                idx_a -= strides_a[j] * (lengths[j] - 1);
                idx_b -= strides_b[j] * (lengths[j] - 1);
            }
        }
    }

    inline void sum_fp32(size_t n, size_t ndim, const size_t *lengths, const bool *mask, float *dst, const float *src) {
        size_t ndim_buf[2][NDIM_STACK_BUF_SIZE];

        workspace ndim_workspace(device_type::cpu);
        size_t *strides_dst, *coord;
        if (ndim > NDIM_STACK_BUF_SIZE) {
            ndim_workspace.init(sizeof(size_t) * ndim * 2);
            strides_dst = static_cast<size_t *>(ndim_workspace);
            coord = static_cast<size_t *>(ndim_workspace) + ndim;
        }
        else {
            strides_dst = ndim_buf[0];
            coord = ndim_buf[1];
        }
        size_t m = calc_strides(strides_dst, ndim, lengths, mask);
        memset(dst, 0, m * sizeof(float));

        memset(coord, 0, ndim * sizeof(size_t));
        size_t idx_dst = 0;
        for (size_t i = 0; i < n; i++) {
            // todo simd
            dst[idx_dst] += src[i];
            for (size_t j = 0; j < ndim; ++j) {
                coord[j]++;
                if (coord[j] < lengths[j]) {
                    idx_dst += strides_dst[j];
                    break;
                }
                // else: coordination carry
                coord[j] = 0;
                idx_dst -= strides_dst[j] * (lengths[j] - 1);
            }
        }
    }

    inline void softmax_fp32(size_t blk_n, size_t blk_len, float *dst, const float *src) {
        for (size_t i = 0; i < blk_n; i++) {
            float *dst_local = dst + i * blk_len;
            const float *src_local = src + i * blk_len;

            float max_val = src_local[0];
            for (size_t j = 0; j < blk_len; j++)
                max_val = std::max(max_val, src_local[j]);

            float sum = 0.0f;
            for (size_t j = 0; j < blk_len; j++) {
                dst_local[j] = expf(src_local[j] - max_val);
                sum += dst_local[j];
            }

            float inv_sum = 1.0f / sum;
            for (size_t j = 0; j < blk_len; j++)
                dst_local[j] *= inv_sum;
        }
    }

    inline void log_softmax_fp32(size_t blk_n, size_t blk_len, float *dst, const float *src) {
        for (size_t i = 0; i < blk_n; i++) {
            float *dst_local = dst + i * blk_len;
            const float *src_local = src + i * blk_len;

            float max_val = src_local[0];
            for (size_t j = 0; j < blk_len; j++)
                max_val = std::max(max_val, src_local[j]);

            float sum = 0.0f;
            for (size_t j = 0; j < blk_len; j++)
                sum += expf(src_local[j] - max_val);

            float offset = max_val + logf(sum);
            for (size_t j = 0; j < blk_len; j++)
                dst_local[j] = src_local[j] - offset;
        }
    }

    inline void correct_count_fp32(size_t blk_n, size_t blk_len, size_t *ret, const float *out, const float *ans) {
        size_t count = 0;
        for (size_t i = 0; i < blk_n; i++) {
            size_t predicted = 0;
            const float *out_local = out + i * blk_len;
            for (size_t j = 1; j < blk_len; j++)
                if (out_local[j] > out_local[predicted])
                    predicted = j;
            if (ans[i * blk_len + predicted] > 0.5f) // 0.0f or 1.0f
                count++;
        }
        *ret = count;
    }

    inline void maxpool_fp32(size_t channel_n, size_t h, size_t w, size_t h_stride, size_t w_stride,
                             float *dst, bool *mask, const float *src) {
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
                                      float *dst, const bool *mask, const float *src) {
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
