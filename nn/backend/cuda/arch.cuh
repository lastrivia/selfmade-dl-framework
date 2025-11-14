#pragma once

#include "except.h"
#include "backend/cpu/ndim.h"

namespace cuda_backend {

    static constexpr size_t THREADS_PER_BLOCK = 256;

    class CudaStreamHandle {
    public:
        CudaStreamHandle() {
            cudaError_t err = cudaStreamCreate(&stream_);
            if (err != cudaSuccess) {
                throw FatalExcept("cuda stream creation failed", __FILE__, __LINE__);
            }
        }

        ~CudaStreamHandle() {
            if (stream_)
                cudaStreamDestroy(stream_);
        }

        CudaStreamHandle(const CudaStreamHandle &) = delete;
        CudaStreamHandle &operator=(const CudaStreamHandle &) = delete;

        CudaStreamHandle(CudaStreamHandle &&other) noexcept : stream_(other.stream_) {
            other.stream_ = nullptr;
        }

        CudaStreamHandle &operator=(CudaStreamHandle &&other) noexcept {
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
        static CudaStreamHandle stream;
        return stream.get();
    }

    static constexpr size_t NDIM_STACK_BUF_SIZE = cpu_backend::NDIM_STACK_BUF_SIZE,
                            NDIM_STACK_BUF_ELEMENTS = 10;
    using cpu_backend::calc_strides;

    class NDimDeviceBuf {
    public:
        NDimDeviceBuf(size_t elements) {
            cudaError_t err = cudaMallocAsync(&buf, elements * NDIM_STACK_BUF_SIZE * sizeof(size_t), default_stream());
            if (err != cudaSuccess)
                throw FatalExcept("cuda memory allocation failed", __FILE__, __LINE__);
        }

        ~NDimDeviceBuf() {
            cudaFree(buf);
        }

        size_t *operator[](size_t index) const {
            return buf + index * NDIM_STACK_BUF_SIZE;
        }

    private:
        size_t *buf;

    };

    static NDimDeviceBuf ndim_device_buf(NDIM_STACK_BUF_ELEMENTS);
}
