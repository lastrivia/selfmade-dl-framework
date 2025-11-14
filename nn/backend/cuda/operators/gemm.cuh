#pragma once

#include <cublas_v2.h>

#include "except.h"
#include "../arch.cuh"

namespace cuda_backend {

    class CublasHandle {
    public:
        CublasHandle() {
            cublasStatus_t status = cublasCreate_v2(&handle_);
            if (status != CUBLAS_STATUS_SUCCESS)
                throw FatalExcept("cublas handle creation failed", __FILE__, __LINE__);
            status = cublasSetStream_v2(handle_, default_stream());
            if (status != CUBLAS_STATUS_SUCCESS)
                throw FatalExcept("cublas handle creation failed", __FILE__, __LINE__);
        }

        ~CublasHandle() {
            if (handle_)
                cublasDestroy_v2(handle_);
        }

        CublasHandle(const CublasHandle &) = delete;
        CublasHandle &operator=(const CublasHandle &) = delete;

        CublasHandle(CublasHandle &&other) noexcept : handle_(other.handle_) {
            other.handle_ = nullptr;
        }

        CublasHandle &operator=(CublasHandle &&other) noexcept {
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
        static CublasHandle handle;
        return handle.get();
    }

    template<bool transpose_a, bool transpose_b>
    void gemm_fp32(size_t m, size_t p, size_t n, float *dst, const float *src_a, const float *src_b) {

        // C: row-major, CUBLAS: col-major
        // memory(A, row-major) == memory(A^T, col-major)

        float alpha = 1.0f, beta = 0.0f;

        // for C-style C(m*n) = A(m*p)B(p*n), invoke CUBLAS C^T(n*m) = B^T(n*p)A^T(p*m)

        cublasStatus_t status = cublasSgemm_v2_64(default_cublas_handle(),
                                                  transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                  transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                  n, m, p,
                                                  &alpha,
                                                  src_b, transpose_b ? p : n,
                                                  src_a, transpose_a ? m : p,
                                                  &beta,
                                                  dst, n);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw FatalExcept("cublas gemm failed", __FILE__, __LINE__);
    }

}
