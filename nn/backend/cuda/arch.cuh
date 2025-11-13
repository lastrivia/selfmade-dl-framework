#pragma once

#include "except.h"
#include "backend/cpu/ndim.h"

namespace cuda_backend {

    static constexpr size_t THREADS_PER_BLOCK = 256;

    class cuda_stream {
    public:
        cuda_stream() {
            cudaError_t err = cudaStreamCreate(&stream_);
            if (err != cudaSuccess) {
                throw nn_except("cuda stream creation failed", __FILE__, __LINE__);
            }
        }

        ~cuda_stream() {
            if (stream_)
                cudaStreamDestroy(stream_);
        }

        cuda_stream(const cuda_stream &) = delete;
        cuda_stream &operator=(const cuda_stream &) = delete;

        cuda_stream(cuda_stream &&other) noexcept : stream_(other.stream_) {
            other.stream_ = nullptr;
        }

        cuda_stream &operator=(cuda_stream &&other) noexcept {
            if (this != &other) {
                if (stream_)
                    cudaStreamDestroy(stream_);
                stream_ = other.stream_;
                other.stream_ = nullptr;
            }
            return *this;
        }

        cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_;
    };

    inline cudaStream_t default_stream() {
        static cuda_stream stream;
        return stream.get();
    }

    static constexpr size_t NDIM_STACK_BUF_SIZE = cpu_backend::NDIM_STACK_BUF_SIZE,
                            NDIM_STACK_BUF_ELEMENTS = 10;
    using cpu_backend::calc_strides;

    class ndim_device_buf_t {
    public:
        ndim_device_buf_t(size_t elements) {
            cudaError_t err = cudaMallocAsync(&buf, elements * NDIM_STACK_BUF_SIZE * sizeof(size_t), default_stream());
            if (err != cudaSuccess)
                throw nn_except("cuda memory allocation failed", __FILE__, __LINE__);
        }

        ~ndim_device_buf_t() {
            cudaFree(buf);
        }

        size_t *operator[](size_t index) const {
            return buf + index * NDIM_STACK_BUF_SIZE;
        }

    private:
        size_t *buf;

    };

    static ndim_device_buf_t ndim_device_buf(NDIM_STACK_BUF_ELEMENTS);
}
