#pragma once

#include "tensor_impl.h"
#include "autograd.h"

#include "interface_impl.generated.h"


inline Tensor flatten(const Tensor &src) {
    // temporary, todo use tensor_view to replace this

    switch (src->dtype_) {
    case ScalarType::fp32: {
        Tensor result({src->shape_.lengths.back(), src->shape_.size / src->shape_.lengths.back()}, src->device_, src->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "copy(size, result, src)"
        dispatch_kernel(device).copy_fp32(result->shape_.size, result->data_, src->data_);

        if ((src->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeCopyFp32(result, src);
        }

        return result;
    }
    case ScalarType::int32: {
        Tensor result({src->shape_.lengths.back(), src->shape_.size / src->shape_.lengths.back()}, src->device_, src->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "copy(size, result, src)"
        dispatch_kernel(device).copy_int32(result->shape_.size, result->data_, src->data_);

        if ((src->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeCopyFp32(result, src);
        }

        return result;
    }
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

template<typename T>
void Tensor::fill(T x) {
    switch (object_->dtype_) {
    case ScalarType::fp32:
        dispatch_kernel(object_->device_).broadcast_fp32(object_->shape_.size, object_->data_, static_cast<float>(x));
        break;
    case ScalarType::int32:
        dispatch_kernel(object_->device_).broadcast_int32(object_->shape_.size, object_->data_, static_cast<int32_t>(x));
        break;
    }
    object_->version_++;
}

inline Tensor matmul(const Tensor &a, const Tensor &b) {
    return matmul<false, false>(a, b);
}

inline size_t correct_count(const Tensor &logits, const Tensor &label) {
    size_t ret;
    if (logits->device_ != label->device_)
        throw FatalExcept("tensor: operands for correct_count are on different devices", __FILE__, __LINE__);
    if (logits->dtype_ != label->dtype_)
        throw FatalExcept("tensor: operands for correct_count have different data types", __FILE__, __LINE__);

    switch (logits->dtype_) {
    case ScalarType::fp32:
        dispatch_kernel(logits->device_).correct_count_fp32(
            logits->shape_.size / logits->shape_.lengths[0], logits->shape_.lengths[0], &ret, logits->data_, label->data_
        );
        break;
    case ScalarType::int32:
        throw FatalExcept("tensor: correct_count does not support data type int32", __FILE__, __LINE__);
        break;
    }
    return ret;
}
