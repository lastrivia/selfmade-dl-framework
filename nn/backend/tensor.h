#pragma once

#include <cassert>
#include <sstream>
#include <utility>

#include "autograd_base.h"
#include "base_kernel.h"
#include "mem_pool.h"
#include "cuda/arch.cuh"

inline bool tensor_log = false;

class tensor_impl {
    friend class tensor;
    friend class tensor_it;

    // grad nodes
    friend class grad_node;
    friend class grad_engine;
#include "autograd_decl.generated.h"

    friend class nn_optimizer;
    friend class adam_optimizer;
    friend class sgd_optimizer;

public:
    struct shape_t {
        size_t ndim;
        size_t size;
        std::vector<size_t> lengths;

        shape_t() : ndim(0), size(0) {}

        template<typename... Dims>
            requires (sizeof...(Dims) > 0) && (std::convertible_to<Dims, size_t> && ...)
        shape_t(Dims... dims) { // NOLINT(*-explicit-constructor)
            ndim = sizeof...(dims);

            lengths.resize(ndim);
            size_t j = ndim;
            ((lengths[--j] = dims), ...);
            size = calc_size();
        }

        shape_t(std::initializer_list<size_t> dims) : ndim(dims.size()) {
            std::vector<size_t> tmp(dims);
            lengths.resize(ndim);
            for (size_t i = 0; i < ndim; i++)
                lengths[i] = tmp[ndim - i - 1];
            size = calc_size();
        }

        shape_t(size_t ndim, const size_t *lengths) {
            this->ndim = ndim;
            this->lengths.resize(ndim);
            for (size_t i = 0; i < ndim; i++)
                this->lengths[i] = lengths[i];
            size = calc_size();
        }

        size_t calc_size() const {
            size_t result = 1;
            for (size_t i = 0; i < ndim; i++) {
                if (lengths[i] == 0)
                    throw nn_except("tensor dim length cannot be 0", __FILE__, __LINE__);
                size_t size_next = result * lengths[i];
                if (size_next / lengths[i] != result)
                    throw nn_except("tensor size overflow", __FILE__, __LINE__);
                result = size_next;
            }
            return result;
        }

        shape_t(const shape_t &shape) = default;
        shape_t &operator=(const shape_t &shape) = default;
        shape_t(shape_t &&shape) noexcept = default;
        shape_t &operator=(shape_t &&shape) noexcept = default;

        bool operator==(const shape_t &other) const {
            if (size != other.size)
                return false;
            size_t min_dim = std::min(ndim, other.ndim);
            for (size_t i = 0; i < min_dim; i++) {
                if (lengths[i] != other.lengths[i])
                    return false;
            }
            return true;
        }

        bool operator!=(const shape_t &other) const { return !(*this == other); }

        explicit operator std::string() const {
            std::stringstream ss;
            ss << '[';
            for (size_t i = ndim - 1; true; i--) {
                ss << lengths[i];
                if (i > 0)
                    ss << ", ";
                else {
                    ss << ']';
                    break;
                }
            }
            return ss.str();
        }
    };

private:
    explicit tensor_impl(device_desc device = {device_type::cpu}, data_type dtype = data_type::fp32) :
        device_(device), dtype_(dtype), ref_count_(0), version_(0), grad_node_(nullptr), requires_grad_(false) {
        data_ = nullptr;
        if (tensor_log)
            std::cout << "<TENSOR> NEW " << std::string(shape_) << std::endl;
    }

    explicit tensor_impl(const shape_t &shape, device_desc device = {device_type::cpu}, data_type dtype = data_type::fp32) :
        shape_(shape), device_(device), dtype_(dtype),
        ref_count_(0), version_(0), grad_node_(nullptr), requires_grad_(false) {
        data_ = alloc_data(nullptr);
        if (tensor_log)
            std::cout << "<TENSOR> NEW " << std::string(shape_) << std::endl;
    }

    explicit tensor_impl(shape_t &&shape, device_desc device = {device_type::cpu}, data_type dtype = data_type::fp32) :
        shape_(std::move(shape)), device_(device), dtype_(dtype),
        ref_count_(0), version_(0), grad_node_(nullptr), requires_grad_(false) {
        data_ = alloc_data(nullptr);
        if (tensor_log)
            std::cout << "<TENSOR> NEW " << std::string(shape_) << std::endl;
    }

    tensor_impl(const tensor_impl &other) :
        shape_(other.shape_), device_(other.device_), dtype_(other.dtype_),
        ref_count_(0), version_(0), grad_node_(nullptr), requires_grad_(false) {
        data_ = alloc_data(other.data_);
        if (tensor_log)
            std::cout << "<TENSOR> NEW " << std::string(shape_) << std::endl;
    }

public:
    ~tensor_impl() {
        if (tensor_log)
            std::cout << "<TENSOR> DELETE " << std::string(shape_) << std::endl;
        if (data_ != nullptr)
            release_data(data_);
        if (grad_node_ != nullptr)
            delete grad_node_;
        if (grad_data_ != nullptr)
            release_data(grad_data_);
    }

    void requires_grad(bool mode) {
        // set leaf tensors
        if (grad_node_)
            throw nn_except("tensor: cannot modify requires_grad attribution of an internal result", __FILE__, __LINE__);

        if (mode == requires_grad_)
            return;
        if (mode) { // enable grad
            requires_grad_ = true;
            grad_data_ = alloc_data(nullptr);
            zero_grad();
        }
        else { // disable grad
            requires_grad_ = false;
            release_data(grad_data_);
            grad_data_ = nullptr;
        }
    }

    void zero_grad() {
        if (grad_data_ == nullptr)
            return;
        switch (device_.type) {
        case device_type::cpu:
            switch (dtype_) {
            case data_type::fp32:
                memset(grad_data_, 0, sizeof(float) * shape_.size);
                break;
            case data_type::int32:
                memset(grad_data_, 0, sizeof(int32_t) * shape_.size);
                break;
            }
            break;
        case device_type::cuda:
            switch (dtype_) {
            case data_type::fp32:
                cudaMemsetAsync(grad_data_, 0, sizeof(float) * shape_.size, cuda_kernel::default_stream());
                break;
            case data_type::int32:
                cudaMemsetAsync(grad_data_, 0, sizeof(int32_t) * shape_.size, cuda_kernel::default_stream());
                break;
            }
            break;
        }
    }

    shape_t shape() const { return shape_; }
    device_desc device() const { return device_; }
    // layout_t layout() const { return {shape_, device_, dtype_}; }

#include "interface_decl.generated.h"
    friend tensor flatten(const tensor &src);
    friend size_t correct_count(const tensor &logits, const tensor &label);

private:
    shape_t shape_;
    const device_desc device_;
    const data_type dtype_;

    data_ptr data_;

    size_t ref_count_; // count of handles pointing to this
    size_t version_; // only count impl-level in-place operations

    grad_node *grad_node_;
    bool requires_grad_;
    data_ptr grad_data_;
    /* leaf tensors that requires grad:
     *     owns grad_data
     *     grad_node is nullptr
     * intermediate tensors that requires grad:
     *     owns grad_node
     *     grad_data allocated & recycled by grad_engine
     */

    data_ptr alloc_data(data_ptr copy_from) const {
        data_ptr ret;
        switch (device_.type) {
        case device_type::cpu:
            switch (dtype_) {
            case data_type::fp32:
                ret = mem_pool::alloc<float>(shape_.size);
                if (copy_from)
                    memcpy(ret, copy_from, sizeof(float) * shape_.size);
                break;
            case data_type::int32:
                ret = mem_pool::alloc<int32_t>(shape_.size);
                if (copy_from)
                    memcpy(ret, copy_from, sizeof(int32_t) * shape_.size);
                break;
            }
            break;
        case device_type::cuda:
            switch (dtype_) {
            case data_type::fp32:
                ret = cuda_mem_pool::alloc<float>(shape_.size);
                if (copy_from)
                    cudaMemcpyAsync(ret, copy_from, sizeof(float) * shape_.size,
                                    cudaMemcpyDeviceToDevice, cuda_kernel::default_stream());
                break;
            case data_type::int32:
                ret = cuda_mem_pool::alloc<int32_t>(shape_.size);
                if (copy_from)
                    cudaMemcpyAsync(ret, copy_from, sizeof(int32_t) * shape_.size,
                                    cudaMemcpyDeviceToDevice, cuda_kernel::default_stream());
                break;
            }
            break;
        }
        return ret;
    }

    void release_data(data_ptr data) const {
        switch (device_.type) {
        case device_type::cpu:
            mem_pool::recycle(data);
            break;
        case device_type::cuda:
            cuda_mem_pool::recycle(data);
            break;
        }
    }
};

using tensor_shape = tensor_impl::shape_t;

class tensor_it {
    // todo optimize
public:
    tensor_it(tensor_impl *tensor, size_t offset) :
        tensor_(tensor), offset_(offset) {}

    template<typename T>
    tensor_it &operator=(T x) {
        switch (tensor_->dtype_) {
        case data_type::fp32:
            static_cast<float *>(tensor_->data_)[offset_] = static_cast<float>(x);
            break;
        case data_type::int32:
            static_cast<int32_t *>(tensor_->data_)[offset_] = static_cast<int32_t>(x);
            break;
        default:
            throw nn_except("unknown data type", __FILE__, __LINE__);
        }
        tensor_->version_++;
        return *this;
    }

    template<typename T>
    operator T() {
        switch (tensor_->dtype_) {
        case data_type::fp32:
            return static_cast<T>(static_cast<float *>(tensor_->data_)[offset_]);
        case data_type::int32:
            return static_cast<T>(static_cast<int32_t *>(tensor_->data_)[offset_]);
        default:
            throw nn_except("unknown data type", __FILE__, __LINE__);
        }
    }

private:
    tensor_impl *tensor_;
    size_t offset_;
};

class tensor {

    // grad nodes
    friend class grad_node;
    friend class grad_engine;
#include "autograd_decl.generated.h"

public:
    // creating new object

    explicit tensor(device_desc device = {device_type::cpu}, data_type dtype = data_type::fp32) :
        object_(new tensor_impl(device, dtype)) {
        object_->ref_count_++;
    }

    explicit tensor(const tensor_impl::shape_t &shape, device_desc device = {device_type::cpu}, data_type dtype = data_type::fp32) :
        object_(new tensor_impl(shape, device, dtype)) {
        object_->ref_count_++;
    }

    explicit tensor(tensor_impl::shape_t &&shape, device_desc device = {device_type::cpu}, data_type dtype = data_type::fp32) :
        object_(new tensor_impl(std::move(shape), device, dtype)) {
        object_->ref_count_++;
    }

    explicit tensor(std::initializer_list<size_t> shape, device_desc device = {device_type::cpu}, data_type dtype = data_type::fp32) :
        object_(new tensor_impl(tensor_shape(shape), device, dtype)) {
        object_->ref_count_++;
    }

    // reference copy
    tensor(const tensor &other) {
        object_ = other.object_;
        if (object_)
            object_->ref_count_++;
    }

    tensor &operator=(const tensor &other) {
        if (object_ == other.object_ || /* actually implies the former */ this == &other)
            return *this;
        if (object_) {
            object_->ref_count_--;
            if (object_->ref_count_ == 0)
                delete object_;
        }
        object_ = other.object_;
        if (object_)
            object_->ref_count_++;
        return *this;
    }

    tensor(tensor &&other) noexcept {
        object_ = other.object_;
        other.object_ = nullptr;
    }

    tensor &operator=(tensor &&other) noexcept {
        if (object_ == other.object_ || /* actually implies the former */ this == &other)
            return *this;
        if (object_) {
            object_->ref_count_--;
            if (object_->ref_count_ == 0)
                delete object_;
        }
        object_ = other.object_;
        other.object_ = nullptr;
        return *this;
    }

    ~tensor() {
        if (object_) {
            object_->ref_count_--;
            if (object_->ref_count_ == 0)
                delete object_;
        }
    }

    // tensor operator+(const tensor &b) const; // example
#include "interface_decl.generated.h"
    friend tensor flatten(const tensor &src);
    friend size_t correct_count(const tensor &logits, const tensor &label);

    tensor_impl *operator->() const {
        return object_;
    }

    void to_device(device_desc device) {
        // creates a new impl on device, copy data, and switch object pointer to the new impl
        // requires_grad attr of leaf nodes will be copied
        if (device == object_->device_)
            return;

        tensor_impl *new_object = new tensor_impl(object_->shape_, device, object_->dtype_);
        cudaMemcpyKind kind = device.type == device_type::cuda ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        size_t size = 0;
        switch (object_->dtype_) {
        case data_type::fp32:
            size = sizeof(float) * object_->shape_.size;
        case data_type::int32:
            size = sizeof(int32_t) * object_->shape_.size;
        }

        cudaMemcpyAsync(new_object->data_, object_->data_, size, kind, cuda_kernel::default_stream());
        if (object_->requires_grad_ && !object_->grad_node_) { // leaf
            new_object->requires_grad(true);
            // todo save a zero_grad() here
            cudaMemcpyAsync(new_object->grad_data_, object_->grad_data_, size, kind, cuda_kernel::default_stream());
        }
        cudaStreamSynchronize(cuda_kernel::default_stream());

        object_->ref_count_--;
        if (object_->ref_count_ == 0)
            delete object_;
        object_ = new_object;
        object_->ref_count_++;
    }

    template<typename... Dims>
        requires (sizeof...(Dims) > 1) && (std::convertible_to<Dims, size_t> && ...)
    tensor_it at(Dims... dims) {
        constexpr size_t ndim = sizeof...(dims);

        size_t lengths[ndim];
        size_t j = ndim;
        ((lengths[--j] = dims), ...);
        size_t offset = 0, stride = 1;
        if (ndim > object_->shape_.ndim)
            throw nn_except("visited dimension out of bounds", __FILE__, __LINE__);
        for (size_t i = 0; i < ndim; i++) {
            offset += lengths[i] * stride;
            if (lengths[i] > object_->shape_.lengths[i])
                throw nn_except("visited index out of bounds", __FILE__, __LINE__);
            stride *= object_->shape_.lengths[i];
        }
        return {object_, offset};
    }

    tensor_it at(size_t index) {
        if (index >= object_->shape_.size)
            throw nn_except("visited index out of bounds", __FILE__, __LINE__);
        return {object_, index};
    }

    void backward();

    template<typename T>
    void fill(T x);

private:
    tensor_impl *object_;

};

inline grad_node::grad_node(const tensor &result) :
    tensor_(result.object_) {}
