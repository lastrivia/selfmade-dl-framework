#pragma once

#include <unordered_map>

#include <cudnn.h>

#include "../../../except.h"
#include "../../mem_pool.h"
#include "../arch.cuh"

namespace cuda_kernel {

    inline void cudnn_check(const cudnnStatus_t status, const char *file, int line) {
        if (status != CUDNN_STATUS_SUCCESS)
            throw nn_except(std::string("cudnn: ") + cudnnGetErrorString(status), file, line);
    }

#define with_check(status) cudnn_check(status, __FILE__, __LINE__)

    class cudnn_handle {
    public:
        cudnn_handle() { // NOLINT(*-pro-type-member-init)
            with_check(cudnnCreate(&handle_));
            with_check(cudnnSetStream(handle_, default_stream()));
        }

        ~cudnn_handle() {
            if (handle_)
                cudnnDestroy(handle_);
        }

        cudnn_handle(const cudnn_handle &) = delete;
        cudnn_handle &operator=(const cudnn_handle &) = delete;

        cudnn_handle(cudnn_handle &&other) noexcept : handle_(other.handle_) {
            other.handle_ = nullptr;
        }

        cudnn_handle &operator=(cudnn_handle &&other) noexcept {
            if (this != &other) {
                if (handle_)
                    cudnnDestroy(handle_);
                handle_ = other.handle_;
                other.handle_ = nullptr;
            }
            return *this;
        }

        cudnnHandle_t get() const { return handle_; }

    private:
        cudnnHandle_t handle_;
    };

    inline cudnnHandle_t default_cudnn_handle() {
        static cudnn_handle handle;
        return handle.get();
    }

    namespace conv_utils_fp32 {
        struct conv_args {
            size_t n, c_i, c_o, h_in, w_in, h_ker, w_ker, h_pad, w_pad;

            bool operator==(const conv_args &other) const = default;
        };

        inline void hash_combine(std::size_t &seed, std::size_t value) noexcept {
            seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        struct shape_hash {
            size_t operator()(const conv_args &r) const noexcept {
                size_t seed = 0;
                hash_combine(seed, std::hash<size_t>{}(r.n));
                hash_combine(seed, std::hash<size_t>{}(r.c_i));
                hash_combine(seed, std::hash<size_t>{}(r.c_o));
                hash_combine(seed, std::hash<size_t>{}(r.h_in));
                hash_combine(seed, std::hash<size_t>{}(r.w_in));
                hash_combine(seed, std::hash<size_t>{}(r.h_ker));
                hash_combine(seed, std::hash<size_t>{}(r.w_ker));
                hash_combine(seed, std::hash<size_t>{}(r.h_pad));
                hash_combine(seed, std::hash<size_t>{}(r.w_pad));
                return seed;
            }
        };

        class cudnn_tensor_desc {
        public:
            cudnn_tensor_desc(size_t n, size_t c, size_t h, size_t w) { // NOLINT(*-pro-type-member-init)
                if (n > INT32_MAX || c > INT32_MAX || h > INT32_MAX || w > INT32_MAX)
                    throw nn_except("cudnn: parameter out of int32 range", __FILE__, __LINE__);
                with_check(cudnnCreateTensorDescriptor(&desc_));
                with_check(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    static_cast<int>(n), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)));
            }

            ~cudnn_tensor_desc() {
                cudnnDestroyTensorDescriptor(desc_);
            }

            cudnn_tensor_desc(const cudnn_tensor_desc &) = delete;
            cudnn_tensor_desc &operator=(const cudnn_tensor_desc &) = delete;
            cudnn_tensor_desc(cudnn_tensor_desc &&other) = delete;
            cudnn_tensor_desc &operator=(cudnn_tensor_desc &&other) = delete;

            operator cudnnTensorDescriptor_t() const { // NOLINT(*-explicit-constructor)
                return desc_;
            }

        private:
            cudnnTensorDescriptor_t desc_;
        };

        class cudnn_filter_desc {
        public:
            cudnn_filter_desc(size_t n, size_t c, size_t h, size_t w) { // NOLINT(*-pro-type-member-init)
                if (n > INT32_MAX || c > INT32_MAX || h > INT32_MAX || w > INT32_MAX)
                    throw nn_except("cudnn: parameter out of int32 range", __FILE__, __LINE__);
                with_check(cudnnCreateFilterDescriptor(&desc_));
                with_check(cudnnSetFilter4dDescriptor(desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                    static_cast<int>(n), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)));
            }

            ~cudnn_filter_desc() {
                cudnnDestroyFilterDescriptor(desc_);
            }

            cudnn_filter_desc(const cudnn_filter_desc &) = delete;
            cudnn_filter_desc &operator=(const cudnn_filter_desc &) = delete;
            cudnn_filter_desc(cudnn_filter_desc &&other) = delete;
            cudnn_filter_desc &operator=(cudnn_filter_desc &&other) = delete;

            operator cudnnFilterDescriptor_t() const { // NOLINT(*-explicit-constructor)
                return desc_;
            }

        private:
            cudnnFilterDescriptor_t desc_;
        };

        class cudnn_conv_desc {
        public:
            cudnn_conv_desc(size_t h_pad, size_t w_pad) { // NOLINT(*-pro-type-member-init)
                if (h_pad > INT32_MAX || w_pad > INT32_MAX)
                    throw nn_except("cudnn: parameter out of int32 range", __FILE__, __LINE__);
                with_check(cudnnCreateConvolutionDescriptor(&desc_));
                with_check(cudnnSetConvolution2dDescriptor(desc_, static_cast<int>(h_pad), static_cast<int>(w_pad),
                    1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
            }

            ~cudnn_conv_desc() {
                cudnnDestroyConvolutionDescriptor(desc_);
            }

            cudnn_conv_desc(const cudnn_conv_desc &) = delete;
            cudnn_conv_desc &operator=(const cudnn_conv_desc &) = delete;
            cudnn_conv_desc(cudnn_conv_desc &&other) = delete;
            cudnn_conv_desc &operator=(cudnn_conv_desc &&other) = delete;

            operator cudnnConvolutionDescriptor_t() const { // NOLINT(*-explicit-constructor)
                return desc_;
            }

        private:
            cudnnConvolutionDescriptor_t desc_;
        };

        class cudnn_activation_desc {
        public:
            explicit cudnn_activation_desc(cudnnActivationMode_t mode) { // NOLINT(*-pro-type-member-init)
                with_check(cudnnCreateActivationDescriptor(&desc_));
                with_check(cudnnSetActivationDescriptor(desc_, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
            }

            ~cudnn_activation_desc() {
                cudnnDestroyActivationDescriptor(desc_);
            }

            cudnn_activation_desc(const cudnn_activation_desc &) = delete;
            cudnn_activation_desc &operator=(const cudnn_activation_desc &) = delete;
            cudnn_activation_desc(cudnn_activation_desc &&other) = delete;
            cudnn_activation_desc &operator=(cudnn_activation_desc &&other) = delete;

            operator cudnnActivationDescriptor_t() const { // NOLINT(*-explicit-constructor)
                return desc_;
            }

        private:
            cudnnActivationDescriptor_t desc_;
        };

        class cudnn_workspace {
        public:
            explicit cudnn_workspace(size_t bytes) {
                workspace_ = cuda_mem_pool::alloc<char>(bytes);
            }

            ~cudnn_workspace() {
                cuda_mem_pool::recycle(workspace_);
            }

            cudnn_workspace(const cudnn_workspace &) = delete;
            cudnn_workspace &operator=(const cudnn_workspace &) = delete;
            cudnn_workspace(cudnn_workspace &&other) = delete;
            cudnn_workspace &operator=(cudnn_workspace &&other) = delete;

            operator void *() const {
                return workspace_;
            }

        private:
            void *workspace_;
        };

        // todo replace with cuDNN 9 backend API

        /** === FORWARD ===
         *
         *  x * w + b -> y
         *  in[n, c_i, h_x, w_x] * ker[c_o, c_i, h_k, w_k] + bias -> dst[n, c_o, h_y, w_y]
        */
        inline void conv_fp32(
            const size_t n, const size_t c_i, const size_t c_o,
            const float *__restrict in, const size_t h_in, const size_t w_in,
            const float *__restrict ker, const size_t h_ker, const size_t w_ker,
            const size_t h_pad, const size_t w_pad,
            const float *__restrict bias, float *__restrict dst
        ) {

            thread_local std::unordered_map<conv_args, cudnnConvolutionFwdAlgo_t, shape_hash> perf_results;

            const size_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;

            cudnn_tensor_desc in_desc(n, c_i, h_in, w_in),
                              dst_desc(n, c_o, h_dst, w_dst),
                              bias_desc(1, c_o, 1, 1);
            cudnn_filter_desc ker_desc(c_o, c_i, h_ker, w_ker);
            cudnn_conv_desc conv_desc(h_pad, w_pad);

            cudnnConvolutionFwdAlgo_t algo;
            conv_args args = {n, c_i, c_o, h_in, w_in, h_ker, w_ker, h_pad, w_pad};
            auto it = perf_results.find(args);
            if (it == perf_results.end()) {
                cudnnConvolutionFwdAlgoPerf_t returned_algo_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
                int returned_algo_count;
                with_check(cudnnGetConvolutionForwardAlgorithm_v7(
                    default_cudnn_handle(), in_desc, ker_desc, conv_desc, dst_desc,
                    CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned_algo_count, returned_algo_results
                ));
                algo = returned_algo_results[0].algo;
                perf_results.emplace(args, algo);
            }
            else {
                algo = it->second;
            }

            size_t workspace_size;
            with_check(cudnnGetConvolutionForwardWorkspaceSize(
                default_cudnn_handle(), in_desc, ker_desc, conv_desc, dst_desc, algo, &workspace_size
            ));
            cudnn_workspace workspace(workspace_size);

            float alpha1 = 1.0f, alpha2 = 0.0f;
            cudnn_activation_desc identity(CUDNN_ACTIVATION_IDENTITY);
            with_check(cudnnConvolutionBiasActivationForward(
                default_cudnn_handle(), &alpha1, in_desc, in, ker_desc, ker, conv_desc, algo, workspace,
                workspace_size, &alpha2, dst_desc, dst, bias_desc, bias, identity, dst_desc, dst
            ));
        }

        /** === INPUT GRAD ===
         *
         *  dy * w(rotated) -> dx
         *  in[n, c_o, h_y, w_y] * ker[c_o, c_i, h_k, w_k](rotated) -> dst[n, c_i, h_x, w_x]
         *
         *  argument h_pad = h_ker - forward_h_pad - 1
         */
        inline void conv_input_grad_fp32(
            const size_t n, const size_t c_i, const size_t c_o,
            const float *__restrict in /* dy */, const size_t h_in, const size_t w_in,
            const float *__restrict ker, const size_t h_ker, const size_t w_ker,
            const size_t h_pad, const size_t w_pad,
            float *__restrict dst /* dx */
        ) {

            thread_local std::unordered_map<conv_args, cudnnConvolutionBwdDataAlgo_t, shape_hash> perf_results;

            const size_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;
            const size_t forward_h_pad = h_ker - 1 - h_pad, forward_w_pad = w_ker - 1 - w_pad;

            cudnn_tensor_desc in_desc(n, c_o, h_in, w_in),
                              dst_desc(n, c_i, h_dst, w_dst);
            cudnn_filter_desc ker_desc(c_o, c_i, h_ker, w_ker);
            cudnn_conv_desc conv_desc(forward_h_pad, forward_w_pad);

            cudnnConvolutionBwdDataAlgo_t algo;
            conv_args args = {n, c_i, c_o, h_in, w_in, h_ker, w_ker, h_pad, w_pad};
            auto it = perf_results.find(args);
            if (it == perf_results.end()) {
                cudnnConvolutionBwdDataAlgoPerf_t returned_algo_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
                int returned_algo_count;
                with_check(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                    default_cudnn_handle(), ker_desc, in_desc, conv_desc, dst_desc,
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &returned_algo_count, returned_algo_results
                ));
                algo = returned_algo_results[0].algo;
                perf_results.emplace(args, algo);
            }
            else {
                algo = it->second;
            }

            size_t workspace_size;
            with_check(cudnnGetConvolutionBackwardDataWorkspaceSize(
                default_cudnn_handle(), ker_desc, in_desc, conv_desc, dst_desc, algo, &workspace_size
            ));
            cudnn_workspace workspace(workspace_size);

            float alpha = 1.0f, beta = 0.0f;
            with_check(cudnnConvolutionBackwardData(
                default_cudnn_handle(), &alpha, ker_desc, ker, in_desc, in, conv_desc,
                algo, workspace, workspace_size, &beta, dst_desc, dst
            ));
        }

        /** === KERNEL GRAD ===
         *
         *  x * dy -> dw
         *  in[n, c_i, h_x, w_x] * ker[n, c_o, h_y, w_y] -> dst[c_o, c_i, h_k, w_k]
         */
        inline void conv_kernel_grad_fp32(
            const size_t n, const size_t c_i, const size_t c_o,
            const float *__restrict in, const size_t h_in, const size_t w_in,
            const float *__restrict ker /* dy */, const size_t h_ker, const size_t w_ker,
            const size_t h_pad, const size_t w_pad,
            float *__restrict dst /* dw */
        ) {

            thread_local std::unordered_map<conv_args, cudnnConvolutionBwdFilterAlgo_t, shape_hash> perf_results;

            const size_t h_dst = h_in + h_pad * 2 - h_ker + 1, w_dst = w_in + w_pad * 2 - w_ker + 1;

            cudnn_tensor_desc in_desc(n, c_i, h_in, w_in),
                              ker_desc(n, c_o, h_ker, w_ker);
            cudnn_filter_desc dst_desc(c_o, c_i, h_dst, w_dst);
            cudnn_conv_desc conv_desc(h_pad, w_pad);

            cudnnConvolutionBwdFilterAlgo_t algo;
            conv_args args = {n, c_i, c_o, h_in, w_in, h_ker, w_ker, h_pad, w_pad};
            auto it = perf_results.find(args);
            if (it == perf_results.end()) {
                cudnnConvolutionBwdFilterAlgoPerf_t returned_algo_results[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
                int returned_algo_count;
                with_check(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                    default_cudnn_handle(), in_desc, ker_desc, conv_desc, dst_desc,
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &returned_algo_count, returned_algo_results
                ));
                algo = returned_algo_results[0].algo;
                perf_results.emplace(args, algo);
            }
            else {
                algo = it->second;
            }

            size_t workspace_size;
            with_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                default_cudnn_handle(), in_desc, ker_desc, conv_desc, dst_desc, algo, &workspace_size
            ));
            cudnn_workspace workspace(workspace_size);

            float alpha = 1.0f, beta = 0.0f;
            with_check(cudnnConvolutionBackwardFilter(
                default_cudnn_handle(), &alpha, in_desc, in, ker_desc, ker, conv_desc,
                algo, workspace, workspace_size, &beta, dst_desc, dst
            ));
        }
    }

    using conv_utils_fp32::conv_fp32;
    using conv_utils_fp32::conv_input_grad_fp32;
    using conv_utils_fp32::conv_kernel_grad_fp32;

#undef with_check

}
