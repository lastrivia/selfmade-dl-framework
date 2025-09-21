#pragma once

#include "../arch.cuh"

#include <cublas_v2.h>

namespace cuda_kernel {

    class cublas_handle {
    public:
        cublas_handle() {
            cublasStatus_t status = cublasCreate_v2(&handle_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cublas handle creation failed");
            }
            cublasSetStream_v2(handle_, default_stream());
        }

        ~cublas_handle() {
            if (handle_)
                cublasDestroy_v2(handle_);
        }

        cublas_handle(const cublas_handle &) = delete;
        cublas_handle &operator=(const cublas_handle &) = delete;

        cublas_handle(cublas_handle &&other) noexcept : handle_(other.handle_) {
            other.handle_ = nullptr;
        }

        cublas_handle &operator=(cublas_handle &&other) noexcept {
            if (this != &other) {
                if (handle_)
                    cublasDestroy_v2(handle_);
                handle_ = other.handle_;
                other.handle_ = nullptr;
            }
            return *this;
        }

        cublasHandle_t get() const { return handle_; }

    private:
        cublasHandle_t handle_;
    };

    inline cublasHandle_t default_cublas_handle() {
        static cublas_handle handle;
        return handle.get();
    }

    template<bool transpose_a, bool transpose_b>
    void gemm_fp32(size_t m, size_t p, size_t n, float *dst, const float *src_a, const float *src_b) noexcept {

        // C: row-major, CUBLAS: col-major
        // memory(A, row-major) == memory(A^T, col-major)

        float alpha = 1.0f, beta = 0.0f;

        // for C-style C(m*n) = A(m*p)B(p*n), invoke CUBLAS C^T(n*m) = B^T(n*p)A^T(p*m)
        cublasSgemm_v2_64(default_cublas_handle(),
                          transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                          transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                          n, m, p,
                          &alpha,
                          src_b, transpose_b ? p : n,
                          src_a, transpose_a ? m : p,
                          &beta,
                          dst, n);
    }

}
