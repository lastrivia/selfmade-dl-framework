#pragma once

#include <cmath>

#include "backend/mem_pool.h"
#include "../arch.cuh"

// ReSharper disable CppNonInlineFunctionDefinitionInHeaderFile

namespace cuda_backend {

    template<typename T>
        requires std::is_trivially_copyable_v<T>
    void copy_raw(size_t n, T *dst, const T *src) {
        cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice, default_stream());
    }

    template<typename Fn, typename... Args>
    void launch_common_kernel(Fn fn, size_t n, Args... args) {
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

    template<typename T>
    __global__ void add_ewise_worker(size_t n, T *dst, const T *src_a, const T *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src_a[idx] + src_b[idx];
    }

    template<typename T>
    void add_ewise(size_t n, T *dst, const T *src_a, const T *src_b) {
        launch_common_kernel(add_ewise_worker<T>, n, dst, src_a, src_b);
    }

    __global__ void sub_ewise_fp32_worker(size_t n, float *dst, const float *src_a, const float *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src_a[idx] - src_b[idx];
    }

    inline void sub_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) {
        launch_common_kernel(sub_ewise_fp32_worker, n, dst, src_a, src_b);
    }

    __global__ void mul_ewise_fp32_worker(size_t n, float *dst, const float *src_a, const float *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src_a[idx] * src_b[idx];
    }

    inline void mul_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) {
        launch_common_kernel(mul_ewise_fp32_worker, n, dst, src_a, src_b);
    }

    __global__ void div_ewise_fp32_worker(size_t n, float *dst, const float *src_a, const float *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src_a[idx] / src_b[idx];
    }

    inline void div_ewise_fp32(size_t n, float *dst, const float *src_a, const float *src_b) {
        launch_common_kernel(div_ewise_fp32_worker, n, dst, src_a, src_b);
    }

    __global__ void add_scalar_fp32_worker(size_t n, float *dst, const float *src, float scalar) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] + scalar;
    }

    inline void add_scalar_fp32(size_t n, float *dst, const float *src, float scalar) {
        launch_common_kernel(add_scalar_fp32_worker, n, dst, src, scalar);
    }

    __global__ void mul_scalar_fp32_worker(size_t n, float *dst, const float *src, float scalar) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] * scalar;
    }

    inline void mul_scalar_fp32(size_t n, float *dst, const float *src, float scalar) {
        launch_common_kernel(mul_scalar_fp32_worker, n, dst, src, scalar);
    }

    __global__ void pow_fp32_worker(size_t n, float *dst, const float *src, float scalar) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = powf(src[idx], scalar);
    }

    inline void pow_fp32(size_t n, float *dst, const float *src, float scalar) {
        launch_common_kernel(pow_fp32_worker, n, dst, src, scalar);
    }

    __global__ void broadcast_fp32_worker(size_t n, float *dst, float val) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = val;
    }

    inline void broadcast_fp32(size_t n, float *dst, float val) {
        launch_common_kernel(broadcast_fp32_worker, n, dst, val);
    }

    __global__ void broadcast_int32_worker(size_t n, int32_t *dst, int32_t val) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = val;
    }

    inline void broadcast_int32(size_t n, int32_t *dst, int32_t val) {
        launch_common_kernel(broadcast_int32_worker, n, dst, val);
    }

    __global__ void square_fp32_worker(size_t n, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = src[idx] * src[idx];
    }

    inline void square_fp32(size_t n, float *dst, const float *src) {
        launch_common_kernel(square_fp32_worker, n, dst, src);
    }

    __global__ void sqrt_fp32_worker(size_t n, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = sqrtf(src[idx]);
    }

    inline void sqrt_fp32(size_t n, float *dst, const float *src) {
        launch_common_kernel(sqrt_fp32_worker, n, dst, src);
    }

    __global__ void relu_fp32_worker(size_t n, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = fmaxf(src[idx], 0.0f);
    }

    inline void relu_fp32(size_t n, float *dst, const float *src) {
        launch_common_kernel(relu_fp32_worker, n, dst, src);
    }

    __global__ void relu_backward_fp32_worker(size_t n, float *dst, const float *src, const float *mask) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            dst[idx] = mask[idx] > 0.0f ? src[idx] : 0.0f;
    }

    inline void relu_backward_fp32(size_t n, float *dst, const float *src, const float *mask) {
        launch_common_kernel(relu_backward_fp32_worker, n, dst, src, mask);
    }

    __global__ void add_broadcast_fp32_worker(size_t n, size_t ndim, const size_t *lengths, const size_t *stride_a, const size_t *stride_b,
                                              float *dst, const float *src_a, const float *src_b) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            size_t x = idx, idx_a = 0, idx_b = 0;
            for (size_t i = 0; i < ndim; i++) {
                size_t mod = x % lengths[i];
                x /= lengths[i];
                idx_a += mod * stride_a[i];
                idx_b += mod * stride_b[i];
            }
            dst[idx] = src_a[idx_a] + src_b[idx_b];
        }
    }

    inline void add_broadcast_fp32(size_t n, size_t ndim, const size_t *lengths, const bool *mask_a, const bool *mask_b,
                                   float *dst, const float *src_a, const float *src_b) {
        size_t ndim_host_buf[2][NDIM_STACK_BUF_SIZE];

        Workspace ndim_host_workspace(DeviceType::cpu);
        size_t *strides_a, *strides_b;
        if (ndim > NDIM_STACK_BUF_SIZE) {
            ndim_host_workspace.init(sizeof(size_t) * ndim * 2);
            strides_a = static_cast<size_t *>(ndim_host_workspace);
            strides_b = static_cast<size_t *>(ndim_host_workspace) + ndim;
        }
        else {
            strides_a = ndim_host_buf[0];
            strides_b = ndim_host_buf[1];
        }
        calc_strides(strides_a, ndim, lengths, mask_a);
        calc_strides(strides_b, ndim, lengths, mask_b);

        Workspace ndim_cuda_workspace(DeviceType::cuda);
        size_t *lengths_cuda, *strides_a_cuda, *strides_b_cuda;
        if (ndim > NDIM_STACK_BUF_SIZE) {
            ndim_cuda_workspace.init(sizeof(size_t) * ndim * 3);
            lengths_cuda = static_cast<size_t *>(ndim_cuda_workspace);
            strides_a_cuda = static_cast<size_t *>(ndim_cuda_workspace) + ndim;
            strides_b_cuda = static_cast<size_t *>(ndim_cuda_workspace) + ndim * 2;
        }
        else {
            lengths_cuda = ndim_device_buf[0];
            strides_a_cuda = ndim_device_buf[1];
            strides_b_cuda = ndim_device_buf[2];
        }
        cudaMemcpyAsync(lengths_cuda, lengths, sizeof(size_t) * ndim, cudaMemcpyHostToDevice, default_stream());
        cudaMemcpyAsync(strides_a_cuda, strides_a, sizeof(size_t) * ndim, cudaMemcpyHostToDevice, default_stream());
        cudaMemcpyAsync(strides_b_cuda, strides_b, sizeof(size_t) * ndim, cudaMemcpyHostToDevice, default_stream());

        launch_common_kernel(add_broadcast_fp32_worker, n, ndim, lengths_cuda, strides_a_cuda, strides_b_cuda, dst, src_a, src_b);
        cudaStreamSynchronize(default_stream()); // otherwise *_cuda data might be covered or deleted
    }

    // todo optimize this

    __global__ void sum_fp32_worker(size_t threads_per_dst /* 2^n or n*THREADS_PER_BLOCK */, size_t dst_n, size_t src_per_dst /* n / dst_n */,
                                    size_t ndim, const size_t *lengths, const size_t *strides, const bool *mask,
                                    size_t *coord_shared_buf /* nullptr if ndim < NDIM_STACK_BUF_SIZE */, float *dst, const float *src) {
        size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t idx_dst = global_tid / threads_per_dst;

        if (idx_dst < dst_n) {
            size_t block_tid = threadIdx.x;
            size_t local_tid = block_tid & (threads_per_dst - 1); // bit-trick: block_tid % threads_per_dst;
            size_t src_from = src_per_dst * local_tid / threads_per_dst;
            size_t src_n = src_per_dst * (local_tid + 1) / threads_per_dst - src_from;

            size_t idx_src = 0;

            size_t x = idx_dst;
            for (size_t i = 0; i < ndim; i++) {
                if (mask[i]) { // keep dimension
                    idx_src += x % lengths[i] * strides[i];
                    x /= lengths[i];
                }
            }

            float sum = 0;
            size_t coord_buf[NDIM_STACK_BUF_SIZE];
            size_t *coord = ndim > NDIM_STACK_BUF_SIZE ? coord_shared_buf + global_tid * ndim : coord_buf;

            x = src_from;
            for (size_t i = 0; i < ndim; i++) {
                if (!mask[i]) { // reduce dimension
                    coord[i] = x % lengths[i];
                    x /= lengths[i];
                    idx_src += coord[i] * strides[i];
                }
            }

            for (size_t i = 0; i < src_n; i++) {
                sum += src[idx_src];
                for (size_t j = 0; j < ndim; j++) {
                    if (!mask[j]) { // reduce dimension
                        coord[j]++;
                        if (coord[j] < lengths[j]) {
                            idx_src += strides[j];
                            break;
                        }
                        // else: coordination carry
                        coord[j] = 0;
                        idx_src -= strides[j] * (lengths[j] - 1);
                    }
                }
            }

            extern __shared__ float shared_mem[];
            shared_mem[block_tid] = sum;
            __syncthreads();

            for (size_t offset = threads_per_dst >> 1; offset; offset >>= 1) {
                if (local_tid < offset)
                    shared_mem[block_tid] += shared_mem[block_tid + offset];
                __syncthreads();
            }

            if (local_tid == 0)
                dst[idx_dst] = shared_mem[block_tid];
        }
    }

    inline size_t suggested_threads_per_dst(size_t src_per_dst) { /* 2^n or n*THREADS_PER_BLOCK */
        static constexpr size_t MAX_SRC_PER_THREAD = 32;
        size_t threads = (src_per_dst + MAX_SRC_PER_THREAD - 1) / MAX_SRC_PER_THREAD;
        size_t pow2 = 1;
        while (pow2 < threads && pow2 < THREADS_PER_BLOCK)
            pow2 <<= 1;
        return pow2;
    }

    inline void sum_fp32(size_t n, size_t ndim, const size_t *lengths, const bool *mask, float *dst, const float *src) {
        size_t ndim_host_buf[1][NDIM_STACK_BUF_SIZE];

        Workspace ndim_host_workspace(DeviceType::cpu);
        size_t *strides;
        if (ndim > NDIM_STACK_BUF_SIZE) {
            ndim_host_workspace.init(sizeof(size_t) * ndim);
            strides = static_cast<size_t *>(ndim_host_workspace);
        }
        else {
            strides = ndim_host_buf[0];
        }
        size_t dst_n = 1, src_per_dst = 1, x = 1;
        for (size_t i = 0; i < ndim; i++) {
            strides[i] = x;
            x *= lengths[i];
            if (mask[i])
                dst_n *= lengths[i];
            else
                src_per_dst *= lengths[i];
        }

        Workspace ndim_cuda_workspace(DeviceType::cuda);
        size_t *lengths_cuda, *strides_cuda, *coord_shared_buf;
        bool *mask_cuda;
        if (ndim > NDIM_STACK_BUF_SIZE) {
            ndim_cuda_workspace.init(sizeof(size_t) * ndim * 2 + sizeof(bool) * ndim + sizeof(size_t) * ndim * dst_n);
            lengths_cuda = static_cast<size_t *>(ndim_cuda_workspace);
            strides_cuda = lengths_cuda + ndim;
            mask_cuda = reinterpret_cast<bool *>(strides_cuda + ndim);
            coord_shared_buf = reinterpret_cast<size_t *>(mask_cuda + ndim);
        }
        else {
            lengths_cuda = ndim_device_buf[0];
            strides_cuda = ndim_device_buf[1];
            mask_cuda = reinterpret_cast<bool *>(ndim_device_buf[2]);
            coord_shared_buf = nullptr; // use in-kernel register buf
        }

        cudaMemcpyAsync(lengths_cuda, lengths, sizeof(size_t) * ndim, cudaMemcpyHostToDevice, default_stream());
        cudaMemcpyAsync(strides_cuda, strides, sizeof(size_t) * ndim, cudaMemcpyHostToDevice, default_stream());
        cudaMemcpyAsync(mask_cuda, mask, sizeof(bool) * ndim, cudaMemcpyHostToDevice, default_stream());

        size_t threads_per_dst = suggested_threads_per_dst(src_per_dst);
        size_t threads = threads_per_dst * dst_n;
        size_t blocks = (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        sum_fp32_worker<<<
                static_cast<unsigned int>(blocks),
                static_cast<unsigned int>(THREADS_PER_BLOCK),
                sizeof(float) * static_cast<unsigned int>(THREADS_PER_BLOCK),
                default_stream()
                >>>
                (threads_per_dst, dst_n, src_per_dst, ndim, lengths_cuda, strides_cuda, mask_cuda, coord_shared_buf, dst, src);

        // launch_common_kernel(sum_fp32_worker, dst_n, local_n, ndim, lengths_cuda, strides_cuda, mask_cuda, coord_shared_buf, dst, src);
        cudaStreamSynchronize(default_stream());
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

    inline void softmax_fp32(size_t group_n, size_t group_size, float *dst, const float *src) {
        launch_common_kernel(softmax_fp32_worker, group_n, group_size, dst, src);
    }

    __global__ void log_softmax_fp32_worker(size_t group_n, size_t group_size, float *dst, const float *src) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < group_n) {
            float *dst_local = dst + idx * group_size;
            const float *src_local = src + idx * group_size;

            float max_val = src_local[0];
            for (size_t j = 0; j < group_size; j++)
                max_val = cu_max(max_val, src_local[j]);

            float sum = 0.0f;
            for (size_t j = 0; j < group_size; j++) {
                sum += expf(src_local[j] - max_val);
            }

            float offset = max_val + logf(sum);
            for (size_t j = 0; j < group_size; j++)
                dst_local[j] = src_local[j] - offset;
        }
    }

    inline void log_softmax_fp32(size_t group_n, size_t group_size, float *dst, const float *src) {
        launch_common_kernel(log_softmax_fp32_worker, group_n, group_size, dst, src);
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
                                   size_t *ret /* host */, const float *out, const float *ans) {
        Workspace flags_workspace(blk_n * sizeof(bool), DeviceType::cpu);
        Workspace flags_cuda_workspace(blk_n * sizeof(bool), DeviceType::cuda);
        bool *flags = flags_workspace, *flags_cuda = flags_cuda_workspace;
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
                             float *dst, bool *mask, const float *src) {
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
                                      float *dst, const bool *mask, const float *src) {
        size_t h_src = (h - 1) / h_stride + 1;
        size_t w_src = (w - 1) / w_stride + 1;
        launch_common_kernel(maxpool_backward_fp32_worker, channel_n * h_src * w_src, h, w,
                             h_src, w_src, h_stride, w_stride,
                             dst, mask, src);
    }
}
