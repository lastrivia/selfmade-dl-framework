#pragma once

#include <cmath>

#include "../arch.cuh"

namespace cuda_kernel {

    template<typename Fn, typename... Args>
    void launch_common_kernel(Fn fn, size_t n, Args... args) noexcept {
        size_t blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        fn<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(THREADS_PER_BLOCK), 0, default_stream()>>>(n, args...);
    }

    template<typename T>
    __device__ __forceinline__ T cu_min(T a, T b) {
        return a < b ? a : b;
    }

    template<typename T>
    __device__ __forceinline__ T cu_max(T a, T b) {
        return a > b ? a : b;
    }

    __global__ void add_ewise_fp32_worker(size_t n, float *dst, const float *src_a, const float *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src_a[idx] + src_b[idx];
    }

    inline void add_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) noexcept {
        launch_common_kernel(add_ewise_fp32_worker, n, dst, src_a, src_b);
    }

    __global__ void sub_ewise_fp32_worker(size_t n, float *dst, const float *src_a, const float *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src_a[idx] - src_b[idx];
    }

    inline void sub_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) noexcept {
        launch_common_kernel(sub_ewise_fp32_worker, n, dst, src_a, src_b);
    }

    __global__ void mul_ewise_fp32_worker(size_t n, float *dst, const float *src_a, const float *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src_a[idx] * src_b[idx];
    }

    inline void mul_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) noexcept {
        launch_common_kernel(mul_ewise_fp32_worker, n, dst, src_a, src_b);
    }

    __global__ void div_ewise_fp32_worker(size_t n, float *dst, const float *src_a, const float *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src_a[idx] / src_b[idx];
    }

    inline void div_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) noexcept {
        launch_common_kernel(div_ewise_fp32_worker, n, dst, src_a, src_b);
    }

    __global__ void add_scalar_fp32_worker(size_t n, float *dst, const float *src, float scalar) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] + scalar;
    }

    inline void add_scalar_fp32(size_t n, float *dst, const float *src, float scalar) noexcept {
        launch_common_kernel(add_scalar_fp32_worker, n, dst, src, scalar);
    }

    __global__ void mul_scalar_fp32_worker(size_t n, float *dst, const float *src, float scalar) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] * scalar;
    }

    inline void mul_scalar_fp32(size_t n, float *dst, const float *src, float scalar) noexcept {
        launch_common_kernel(mul_scalar_fp32_worker, n, dst, src, scalar);
    }

    __global__ void pow_fp32_worker(size_t n, float *dst, const float *src, float scalar) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = powf(src[idx], scalar);
    }

    inline void pow_fp32(size_t n, float *dst, const float *src, float scalar) noexcept {
        launch_common_kernel(pow_fp32_worker, n, dst, src, scalar);
    }

    __global__ void broadcast_fp32_worker(size_t n, float *dst, float val) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = val;
    }

    inline void broadcast_fp32(size_t n, float *dst, float val) noexcept {
        launch_common_kernel(broadcast_fp32_worker, n, dst, val);
    }

    __global__ void square_fp32_worker(size_t n, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] * src[idx];
    }

    inline void square_fp32(size_t n, float *dst, const float *src) noexcept {
        launch_common_kernel(square_fp32_worker, n, dst, src);
    }

    __global__ void sqrt_fp32_worker(size_t n, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = sqrtf(src[idx]);
    }

    inline void sqrt_fp32(size_t n, float *dst, const float *src) noexcept {
        launch_common_kernel(sqrt_fp32_worker, n, dst, src);
    }

    __global__ void relu_fp32_worker(size_t n, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = fmaxf(src[idx], 0.0f);
    }

    inline void relu_fp32(size_t n, float *dst, const float *src) noexcept {
        launch_common_kernel(relu_fp32_worker, n, dst, src);
    }

    __global__ void relu_backward_fp32_worker(size_t n, float *dst, const float *src, const float *mask) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = mask[idx] > 0.0f ? src[idx] : 0.0f;
    }

    inline void relu_backward_fp32(size_t n, float *dst, const float *src, const float *mask) noexcept {
        launch_common_kernel(relu_backward_fp32_worker, n, dst, src, mask);
    }

    __global__ void add_cyclic_fp32_worker(size_t n, size_t group_size,
                                           float *dst, const float *src, const float *tile) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] + tile[idx % group_size];
    }

    inline void add_cyclic_fp32(size_t group_n, size_t group_size,
                                float *dst, const float *src, const float *tile) noexcept {
        launch_common_kernel(add_cyclic_fp32_worker, group_n * group_size, group_size, dst, src, tile);
    }

    __global__ void sub_cyclic_fp32_worker(size_t n, size_t group_size,
                                           float *dst, const float *src, const float *tile) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] - tile[idx % group_size];
    }

    inline void sub_cyclic_fp32(size_t group_n, size_t group_size,
                                float *dst, const float *src, const float *tile) noexcept {
        launch_common_kernel(sub_cyclic_fp32_worker, group_n * group_size, group_size, dst, src, tile);
    }

    __global__ void add_stretched_fp32_worker(size_t n, size_t group_size,
                                              float *dst, const float *src, const float *tile) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] + tile[idx / group_size];
    }

    inline void add_stretched_fp32(size_t group_n, size_t group_size,
                                   float *dst, const float *src, const float *tile) noexcept {
        launch_common_kernel(add_stretched_fp32_worker, group_n * group_size, group_size, dst, src, tile);
    }

    __global__ void sub_stretched_fp32_worker(size_t n, size_t group_size,
                                              float *dst, const float *src, const float *tile) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] - tile[idx / group_size];
    }

    inline void sub_stretched_fp32(size_t group_n, size_t group_size,
                                   float *dst, const float *src, const float *tile) noexcept {
        launch_common_kernel(sub_stretched_fp32_worker, group_n * group_size, group_size, dst, src, tile);
    }

    __global__ void sum_cyclic_fp32_worker(size_t group_size, size_t group_n, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // j
        if (idx < group_size) {
            const float *src_local = src + idx;
            float sum = 0.0f;
            for (size_t i = 0; i < group_n; i++)
                sum += src_local[i * group_size];
            dst[idx] = sum;
        }
    }

    inline void sum_cyclic_fp32(size_t group_n, size_t group_size, float *dst, const float *src) noexcept {
        launch_common_kernel(sum_cyclic_fp32_worker, group_size, group_n, dst, src);
    }

    __global__ void sum_stretched_fp32_worker(size_t group_n, size_t group_size, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // i
        if (idx < group_n) {
            const float *src_local = src + idx * group_size;
            float sum = 0.0f;
            for (size_t j = 0; j < group_size; j++)
                sum += src_local[j];
            dst[idx] = sum;
        }
    }

    inline void sum_stretched_fp32(size_t group_n, size_t group_size, float *dst, const float *src) noexcept {
        launch_common_kernel(sum_stretched_fp32_worker, group_n, group_size, dst, src);
    }

    __global__ void softmax_fp32_worker(size_t group_n, size_t group_size, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < group_n) {
            float *dst_local = dst + idx * group_size;
            const float *src_local = src + idx * group_size;

            float max_val = src_local[0];
            for (size_t j = 0; j < group_size; j++)
                max_val = cu_max(max_val, src_local[j]);

            float sum = 0.0f;
            for (size_t j = 0; j < group_size; j++) {
                dst_local[j] = expf(src_local[j] - max_val);
                sum += dst_local[j];
            }

            float inv_sum = 1.0f / sum;
            for (size_t j = 0; j < group_size; j++)
                dst_local[j] *= inv_sum;
        }
    }

    inline void softmax_fp32(size_t group_n, size_t group_size, float *dst, const float *src) noexcept {
        launch_common_kernel(softmax_fp32_worker, group_n, group_size, dst, src);
    }

    __global__ void correct_count_fp32_worker(size_t blk_n, size_t blk_len,
                                              bool *flag, const float *out, const float *ans) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < blk_n) {
            size_t predicted = 0;
            const float *out_local = out + idx * blk_len;
            for (size_t j = 1; j < blk_len; j++)
                if (out_local[j] > out_local[predicted])
                    predicted = j;
            flag[idx] = ans[idx * blk_len + predicted] > 0.5f; // 0.0f or 1.0f
        }
    }

    inline void correct_count_fp32(size_t blk_n, size_t blk_len,
                                   size_t *ret /* host */, const float *out, const float *ans) noexcept {
        bool *flags = mem_pool<device_type::cpu>::alloc<bool>(blk_n);
        bool *flags_cuda = mem_pool<device_type::cuda>::alloc<bool>(blk_n);
        launch_common_kernel(correct_count_fp32_worker, blk_n, blk_len, flags_cuda, out, ans);
        cudaMemcpyAsync(flags, flags_cuda, blk_n * sizeof(bool), cudaMemcpyDeviceToHost, default_stream());
        cudaStreamSynchronize(default_stream());
        size_t count = 0;
        for (size_t i = 0; i < blk_n; i++)
            count += flags[i] ? 1 : 0;
        *ret = count;
    }

    __global__ void maxpool_fp32_worker(size_t group_n, size_t h, size_t w,
                                        size_t h_out, size_t w_out, size_t h_stride, size_t w_stride,
                                        float *dst, bool *mask, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < group_n) {
            size_t channel = idx / h_out / w_out;
            size_t h_block_idx = (idx / w_out) % h_out;
            size_t w_block_idx = idx % w_out;

            size_t h_block_size = cu_min(h_stride, h - h_block_idx * h_stride);
            size_t w_block_size = cu_min(w_stride, w - w_block_idx * w_stride);

            size_t src_offset = (channel * h + h_block_idx * h_stride) * w + w_block_idx * w_stride;
            const float *src_local = src + src_offset;

            float max_val = src_local[0];
            size_t max_h = 0, max_w = 0;

            for (size_t i = 0; i < h_block_size; i++) {
                for (size_t j = 0; j < w_block_size; j++) {
                    float val = src_local[i * w + j];
                    if (val > max_val) {
                        max_val = val;
                        max_h = i;
                        max_w = j;
                    }
                }
            }

            mask[src_offset + max_h * w + max_w] = true;
            dst[(channel * h_out + h_block_idx) * w_out + w_block_idx] = max_val;
        }
    }

    inline void maxpool_fp32(size_t channel_n, size_t h, size_t w, size_t h_stride, size_t w_stride,
                             float *dst, bool *mask, const float *src) noexcept {
        size_t h_out = (h - 1) / h_stride + 1;
        size_t w_out = (w - 1) / w_stride + 1;
        cudaMemsetAsync(mask, 0, channel_n * h * w * sizeof(bool), default_stream());
        launch_common_kernel(maxpool_fp32_worker, channel_n * h_out * w_out, h, w,
                             h_out, w_out, h_stride, w_stride,
                             dst, mask, src);
    }

    __global__ void maxpool_backward_fp32_worker(size_t group_n, size_t h, size_t w,
                                                 size_t h_src, size_t w_src, size_t h_stride, size_t w_stride,
                                                 float *dst, const bool *mask, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        auto *dst_bits = reinterpret_cast<int32_t *>(dst);
        const auto *src_bits = reinterpret_cast<const int32_t *>(src);
        if (idx < group_n) {
            size_t channel = idx / h_src / w_src;
            size_t h_block_idx = (idx / w_src) % h_src;
            size_t w_block_idx = idx % w_src;

            size_t h_block_size = cu_min(h_stride, h - h_block_idx * h_stride);
            size_t w_block_size = cu_min(w_stride, w - w_block_idx * w_stride);

            size_t dst_offset = (channel * h + h_block_idx * h_stride) * w + w_block_idx * w_stride;
            int32_t *dst_bits_local = dst_bits + dst_offset;
            const bool *mask_local = mask + dst_offset;
            const int32_t src_bits_val = src_bits[(channel * h_src + h_block_idx) * w_src + w_block_idx];

            for (size_t i = 0; i < h_block_size; i++) {
                for (size_t j = 0; j < w_block_size; j++) {
                    // bitmask trick:
                    // dst_local[i * w + j] = mask_local[i * w + j] ? src_val : 0.0f
                    int32_t bitmask = -static_cast<int32_t>(mask_local[i * w + j]);
                    dst_bits_local[i * w + j] = src_bits_val & bitmask;
                }
            }
        }
    }

    inline void maxpool_backward_fp32(size_t channel_n, size_t h, size_t w, size_t h_stride, size_t w_stride,
                                      float *dst, const bool *mask, const float *src) noexcept {
        size_t h_src = (h - 1) / h_stride + 1;
        size_t w_src = (w - 1) / w_stride + 1;
        launch_common_kernel(maxpool_backward_fp32_worker, channel_n * h_src * w_src, h, w,
                             h_src, w_src, h_stride, w_stride,
                             dst, mask, src);
    }
}
