#pragma once

#include "tensor.h"
#include "autograd.h"

#include "interface_impl.generated.h"

// inline tensor tensor::operator+(const tensor &b) const {
//     const tensor &a = *this;
//     if (a->device_ != b->device_)
//         throw nn_except("tensor: operands for operator+ are on different devices", __FILE__, __LINE__);
//     if (a->dtype_ != b->dtype_)
//         throw nn_except("tensor: operands for operator+ have different data types", __FILE__, __LINE__);
//
//     // [codegen] requires shape_equal: a, b
//     if (a->shape_ != b->shape_)
//         throw nn_except(std::string() + "tensor: operator+ cannot handle tensors " + std::string(a->shape_) + " and " + std::string(b->shape_), __FILE__,__LINE__);
//
//     switch (a->dtype_) {
//     case data_type::fp32:
//         tensor result(a->shape_, a->device_, a->dtype_);
//
//         // [codegen] "add_ewise(size, result, a, b)"
//         dispatch_kernel(result->device_).add_ewise_fp32(result->shape_.size, result->data_, a->data_, b->data_);
//
//         if ((a->requires_grad_ || b->requires_grad_) && !global_no_grad) {
//             result->requires_grad_ = true;
//             result->grad_node_ = new grad_node_add_fp32(result, a, b);
//         }
//
//         return result;
//         break;
//     case data_type::int32:
//         throw nn_except("tensor: operator+ does not support data type int32", __FILE__, __LINE__);
//         break;
//     default:
//         throw nn_except("tensor: unknown data type", __FILE__, __LINE__);
//         break;
//     }
// }


inline tensor flatten(const tensor &src) {
    // temporary, todo use tensor_view to replace this

    switch (src->dtype_) {
    case data_type::fp32: {
        tensor result({src->shape_.lengths.back(), src->shape_.size / src->shape_.lengths.back()}, src->device_, src->dtype_);
        device_desc device = result->device_;

        // [codegen] "copy(size, result, src)"
        dispatch_kernel(device).copy_fp32(result->shape_.size, result->data_, src->data_);

        if ((src->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new grad_node_copy_fp32(result, src);
        }

        return result;
    }
    case data_type::int32: {
        tensor result({src->shape_.lengths.back(), src->shape_.size / src->shape_.lengths.back()}, src->device_, src->dtype_);
        device_desc device = result->device_;

        // [codegen] "copy(size, result, src)"
        dispatch_kernel(device).copy_int32(result->shape_.size, result->data_, src->data_);

        if ((src->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new grad_node_copy_int32(result, src);
        }

        return result;
    }
    default:
        throw nn_except("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

template<typename T>
void tensor::fill(T x) {
    switch (object_->dtype_) {
    case data_type::fp32:
        dispatch_kernel(object_->device_).broadcast_fp32(object_->shape_.size, object_->data_, static_cast<float>(x));
        break;
    case data_type::int32:
        dispatch_kernel(object_->device_).broadcast_int32(object_->shape_.size, object_->data_, static_cast<int32_t>(x));
        break;
    }
    object_->version_++;
}

inline tensor matmul(const tensor &a, const tensor &b) {
    return matmul<false, false>(a, b);
}

inline size_t correct_count(const tensor &logits, const tensor &label) {
    size_t ret;
    if (logits->device_ != label->device_)
        throw nn_except("tensor: operands for correct_count are on different devices", __FILE__, __LINE__);
    if (logits->dtype_ != label->dtype_)
        throw nn_except("tensor: operands for correct_count have different data types", __FILE__, __LINE__);

    switch (logits->dtype_) {
    case data_type::fp32:
        dispatch_kernel(logits->device_).correct_count_fp32(
            logits->shape_.size / logits->shape_.lengths[0], logits->shape_.lengths[0], &ret, logits->data_, label->data_
        );
        break;
    case data_type::int32:
        throw nn_except("tensor: correct_count does not support data type int32", __FILE__, __LINE__);
        break;
    }
    return ret;
}
