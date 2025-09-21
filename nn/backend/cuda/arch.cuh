#pragma once

#include <stdexcept>

namespace cuda_kernel {

    static constexpr size_t THREADS_PER_BLOCK = 256;

    class cuda_stream {
    public:
        cuda_stream() {
            cudaError_t err = cudaStreamCreate(&stream_);
            if (err != cudaSuccess) {
                throw std::runtime_error("cuda stream creation failed");
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
}
