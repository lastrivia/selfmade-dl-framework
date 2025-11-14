#pragma once

#include <sstream>

#include "backend.h"
#include "autograd_base.h"

inline bool tensor_log = false;

class TensorImpl {
    friend class Tensor;
    friend class TensorIt;

    // grad nodes
    friend class GradNode;
    friend class GradEngine;
#include "autograd_decl.generated.h"

    friend class Optimizer;
    friend class AdamOptimizer;
    friend class SgdOptimizer;

public:
    struct Shape {
        size_t ndim;
        size_t size;
        std::vector<size_t> lengths;

        Shape() : ndim(0), size(0) {}

        template<typename... Dims>
            requires (sizeof...(Dims) > 0) && (std::convertible_to<Dims, size_t> && ...)
        Shape(Dims... dims) { // NOLINT(*-explicit-constructor)
            ndim = sizeof...(dims);

            lengths.resize(ndim);
            size_t j = ndim;
            ((lengths[--j] = dims), ...);
            size = calc_size();
        }

        Shape(std::initializer_list<size_t> dims) : ndim(dims.size()) {
            std::vector<size_t> tmp(dims);
            lengths.resize(ndim);
            for (size_t i = 0; i < ndim; i++)
                lengths[i] = tmp[ndim - i - 1];
            size = calc_size();
        }

        Shape(size_t ndim, const size_t *lengths) {
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
                    throw FatalExcept("tensor dim length cannot be 0", __FILE__, __LINE__);
                size_t size_next = result * lengths[i];
                if (size_next / lengths[i] != result)
                    throw FatalExcept("tensor size overflow", __FILE__, __LINE__);
                result = size_next;
            }
            return result;
        }

        Shape(const Shape &shape) = default;
        Shape &operator=(const Shape &shape) = default;
        Shape(Shape &&shape) noexcept = default;
        Shape &operator=(Shape &&shape) noexcept = default;

        bool operator==(const Shape &other) const {
            if (size != other.size)
                return false;
            size_t min_dim = std::min(ndim, other.ndim);
            for (size_t i = 0; i < min_dim; i++) {
                if (lengths[i] != other.lengths[i])
                    return false;
            }
            return true;
        }

        bool operator!=(const Shape &other) const { return !(*this == other); }

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
    explicit TensorImpl(DeviceDesc device = {DeviceType::cpu}, ScalarType dtype = ScalarType::fp32) :
        device_(device), dtype_(dtype), ref_count_(0), version_(0), grad_node_(nullptr), requires_grad_(false) {
        data_ = nullptr;
        if (tensor_log)
            std::cout << "<TENSOR> NEW " << std::string(shape_) << std::endl;
    }

    explicit TensorImpl(const Shape &shape, DeviceDesc device = {DeviceType::cpu}, ScalarType dtype = ScalarType::fp32) :
        shape_(shape), device_(device), dtype_(dtype),
        ref_count_(0), version_(0), grad_node_(nullptr), requires_grad_(false) {
        data_ = alloc_data(nullptr);
        if (tensor_log)
            std::cout << "<TENSOR> NEW " << std::string(shape_) << std::endl;
    }

    explicit TensorImpl(Shape &&shape, DeviceDesc device = {DeviceType::cpu}, ScalarType dtype = ScalarType::fp32) :
        shape_(std::move(shape)), device_(device), dtype_(dtype),
        ref_count_(0), version_(0), grad_node_(nullptr), requires_grad_(false) {
        data_ = alloc_data(nullptr);
        if (tensor_log)
            std::cout << "<TENSOR> NEW " << std::string(shape_) << std::endl;
    }

    TensorImpl(const TensorImpl &other) :
        shape_(other.shape_), device_(other.device_), dtype_(other.dtype_),
        ref_count_(0), version_(0), grad_node_(nullptr), requires_grad_(false) {
        data_ = alloc_data(other.data_);
        if (tensor_log)
            std::cout << "<TENSOR> NEW " << std::string(shape_) << std::endl;
    }

public:
    ~TensorImpl() {
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
            throw FatalExcept("tensor: cannot modify requires_grad attribution of an internal result", __FILE__, __LINE__);

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
        case DeviceType::cpu:
            switch (dtype_) {
            case ScalarType::fp32:
                memset(grad_data_, 0, sizeof(float) * shape_.size);
                break;
            case ScalarType::int32:
                memset(grad_data_, 0, sizeof(int32_t) * shape_.size);
                break;
            }
            break;
        case DeviceType::cuda:
            switch (dtype_) {
            case ScalarType::fp32:
                cudaMemsetAsync(grad_data_, 0, sizeof(float) * shape_.size, cuda_backend::default_stream());
                break;
            case ScalarType::int32:
                cudaMemsetAsync(grad_data_, 0, sizeof(int32_t) * shape_.size, cuda_backend::default_stream());
                break;
            }
            break;
        }
    }

    Shape shape() const { return shape_; }
    DeviceDesc device() const { return device_; }
    // layout_t layout() const { return {shape_, device_, dtype_}; }

#include "interface_decl.generated.h"
    friend Tensor flatten(const Tensor &src);
    friend size_t correct_count(const Tensor &logits, const Tensor &label);

private:
    Shape shape_;
    const DeviceDesc device_;
    const ScalarType dtype_;

    Data data_;

    size_t ref_count_; // count of handles pointing to this
    size_t version_; // only count impl-level in-place operations

    GradNode *grad_node_;
    bool requires_grad_;
    Data grad_data_;
    /* leaf tensors that requires grad:
     *     owns grad_data
     *     grad_node is nullptr
     * intermediate tensors that requires grad:
     *     owns grad_node
     *     grad_data allocated & recycled by grad_engine
     */

    Data alloc_data(Data copy_from) const {
        Data ret;
        switch (device_.type) {
        case DeviceType::cpu:
            switch (dtype_) {
            case ScalarType::fp32:
                ret = MemPool::alloc<float>(shape_.size);
                if (copy_from)
                    memcpy(ret, copy_from, sizeof(float) * shape_.size);
                break;
            case ScalarType::int32:
                ret = MemPool::alloc<int32_t>(shape_.size);
                if (copy_from)
                    memcpy(ret, copy_from, sizeof(int32_t) * shape_.size);
                break;
            }
            break;
        case DeviceType::cuda:
            switch (dtype_) {
            case ScalarType::fp32:
                ret = CudaMemPool::alloc<float>(shape_.size);
                if (copy_from)
                    cudaMemcpyAsync(ret, copy_from, sizeof(float) * shape_.size,
                                    cudaMemcpyDeviceToDevice, cuda_backend::default_stream());
                break;
            case ScalarType::int32:
                ret = CudaMemPool::alloc<int32_t>(shape_.size);
                if (copy_from)
                    cudaMemcpyAsync(ret, copy_from, sizeof(int32_t) * shape_.size,
                                    cudaMemcpyDeviceToDevice, cuda_backend::default_stream());
                break;
            }
            break;
        }
        return ret;
    }

    void release_data(Data data) const {
        switch (device_.type) {
        case DeviceType::cpu:
            MemPool::recycle(data);
            break;
        case DeviceType::cuda:
            CudaMemPool::recycle(data);
            break;
        }
    }
};

using tensor_shape = TensorImpl::Shape;

class TensorIt {
    // todo optimize
public:
    TensorIt(TensorImpl *tensor, size_t offset) :
        tensor_(tensor), offset_(offset) {}

    template<typename T>
    TensorIt &operator=(T x) {
        switch (tensor_->dtype_) {
        case ScalarType::fp32:
            static_cast<float *>(tensor_->data_)[offset_] = static_cast<float>(x);
            break;
        case ScalarType::int32:
            static_cast<int32_t *>(tensor_->data_)[offset_] = static_cast<int32_t>(x);
            break;
        default:
            throw FatalExcept("unknown data type", __FILE__, __LINE__);
        }
        tensor_->version_++;
        return *this;
    }

    template<typename T>
    operator T() {
        switch (tensor_->dtype_) {
        case ScalarType::fp32:
            return static_cast<T>(static_cast<float *>(tensor_->data_)[offset_]);
        case ScalarType::int32:
            return static_cast<T>(static_cast<int32_t *>(tensor_->data_)[offset_]);
        default:
            throw FatalExcept("unknown data type", __FILE__, __LINE__);
        }
    }

private:
    TensorImpl *tensor_;
    size_t offset_;
};

class Tensor {

    // grad nodes
    friend class GradNode;
    friend class GradEngine;
#include "autograd_decl.generated.h"

public:
    // creating new object

    explicit Tensor(DeviceDesc device = {DeviceType::cpu}, ScalarType dtype = ScalarType::fp32) :
        object_(new TensorImpl(device, dtype)) {
        object_->ref_count_++;
    }

    explicit Tensor(const TensorImpl::Shape &shape, DeviceDesc device = {DeviceType::cpu}, ScalarType dtype = ScalarType::fp32) :
        object_(new TensorImpl(shape, device, dtype)) {
        object_->ref_count_++;
    }

    explicit Tensor(TensorImpl::Shape &&shape, DeviceDesc device = {DeviceType::cpu}, ScalarType dtype = ScalarType::fp32) :
        object_(new TensorImpl(std::move(shape), device, dtype)) {
        object_->ref_count_++;
    }

    explicit Tensor(std::initializer_list<size_t> shape, DeviceDesc device = {DeviceType::cpu}, ScalarType dtype = ScalarType::fp32) :
        object_(new TensorImpl(tensor_shape(shape), device, dtype)) {
        object_->ref_count_++;
    }

    // reference copy
    Tensor(const Tensor &other) {
        object_ = other.object_;
        if (object_)
            object_->ref_count_++;
    }

    Tensor &operator=(const Tensor &other) {
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

    Tensor(Tensor &&other) noexcept {
        object_ = other.object_;
        other.object_ = nullptr;
    }

    Tensor &operator=(Tensor &&other) noexcept {
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

    ~Tensor() {
        if (object_) {
            object_->ref_count_--;
            if (object_->ref_count_ == 0)
                delete object_;
        }
    }

    // tensor operator+(const tensor &b) const; // example
#include "interface_decl.generated.h"
    friend Tensor flatten(const Tensor &src);
    friend size_t correct_count(const Tensor &logits, const Tensor &label);

    TensorImpl *operator->() const {
        return object_;
    }

    void to_device(DeviceDesc device) {
        // creates a new impl on device, copy data, and switch object pointer to the new impl
        // requires_grad attr of leaf nodes will be copied
        if (device == object_->device_)
            return;

        TensorImpl *new_object = new TensorImpl(object_->shape_, device, object_->dtype_);
        cudaMemcpyKind kind = device.type == DeviceType::cuda ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        size_t size = 0;
        switch (object_->dtype_) {
        case ScalarType::fp32:
            size = sizeof(float) * object_->shape_.size;
        case ScalarType::int32:
            size = sizeof(int32_t) * object_->shape_.size;
        }

        cudaMemcpyAsync(new_object->data_, object_->data_, size, kind, cuda_backend::default_stream());
        if (object_->requires_grad_ && !object_->grad_node_) { // leaf
            new_object->requires_grad(true);
            // todo save a zero_grad() here
            cudaMemcpyAsync(new_object->grad_data_, object_->grad_data_, size, kind, cuda_backend::default_stream());
        }
        cudaStreamSynchronize(cuda_backend::default_stream());

        object_->ref_count_--;
        if (object_->ref_count_ == 0)
            delete object_;
        object_ = new_object;
        object_->ref_count_++;
    }

    template<typename... Dims>
        requires (sizeof...(Dims) > 1) && (std::convertible_to<Dims, size_t> && ...)
    TensorIt at(Dims... dims) {
        constexpr size_t ndim = sizeof...(dims);

        size_t lengths[ndim];
        size_t j = ndim;
        ((lengths[--j] = dims), ...);
        size_t offset = 0, stride = 1;
        if (ndim > object_->shape_.ndim)
            throw FatalExcept("visited dimension out of bounds", __FILE__, __LINE__);
        for (size_t i = 0; i < ndim; i++) {
            offset += lengths[i] * stride;
            if (lengths[i] > object_->shape_.lengths[i])
                throw FatalExcept("visited index out of bounds", __FILE__, __LINE__);
            stride *= object_->shape_.lengths[i];
        }
        return {object_, offset};
    }

    TensorIt at(size_t index) {
        if (index >= object_->shape_.size)
            throw FatalExcept("visited index out of bounds", __FILE__, __LINE__);
        return {object_, index};
    }

    void backward();

    template<typename T>
    void fill(T x);

private:
    TensorImpl *object_;

};

inline GradNode::GradNode(const Tensor &result) :
    tensor_(result.object_) {}
