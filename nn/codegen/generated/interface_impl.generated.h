#pragma once
        
#include "backend.h"
#include "tensor/tensor_impl.h"
#include "tensor/autograd.h"

inline Tensor copy(const Tensor &src) {
    // [codegen] shape: identity

    switch (src->dtype_) {
    case ScalarType::fp32: {
        Tensor result(src->shape_, src->device_, src->dtype_);
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
        Tensor result(src->shape_, src->device_, src->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "copy(size, result, src)"
        dispatch_kernel(device).copy_int32(result->shape_.size, result->data_, src->data_);

        if ((src->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeCopyInt32(result, src);
        }

        return result;
    }
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor Tensor::operator+(const Tensor &b) const {
    const Tensor &a = *this;

    if (a->device_ != b->device_)
        throw FatalExcept("tensor: operands for operator+ are on different devices", __FILE__, __LINE__);
    if (a->dtype_ != b->dtype_)
        throw FatalExcept("tensor: operands for operator+ have different data types", __FILE__, __LINE__);

    // [codegen] shape: ewise
    if (a->shape_ != b->shape_)
        throw FatalExcept(std::string() + "tensor: operator+ cannot handle tensors " + std::string(a->shape_) + " and " + std::string(b->shape_), __FILE__ , __LINE__);

    switch (a->dtype_) {
    case ScalarType::fp32: {
        Tensor result(a->shape_, a->device_, a->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "add_ewise(size, result, a, b)"
        dispatch_kernel(device).add_ewise_fp32(result->shape_.size, result->data_, a->data_, b->data_);

        if ((a->requires_grad_ || b->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeAddFp32(result, a, b);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: operator+ does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor Tensor::operator-(const Tensor &b) const {
    const Tensor &a = *this;

    if (a->device_ != b->device_)
        throw FatalExcept("tensor: operands for operator- are on different devices", __FILE__, __LINE__);
    if (a->dtype_ != b->dtype_)
        throw FatalExcept("tensor: operands for operator- have different data types", __FILE__, __LINE__);

    // [codegen] shape: ewise
    if (a->shape_ != b->shape_)
        throw FatalExcept(std::string() + "tensor: operator- cannot handle tensors " + std::string(a->shape_) + " and " + std::string(b->shape_), __FILE__ , __LINE__);

    switch (a->dtype_) {
    case ScalarType::fp32: {
        Tensor result(a->shape_, a->device_, a->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "sub_ewise(size, result, a, b)"
        dispatch_kernel(device).sub_ewise_fp32(result->shape_.size, result->data_, a->data_, b->data_);

        if ((a->requires_grad_ || b->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeSubFp32(result, a, b);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: operator- does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor Tensor::operator*(const Tensor &b) const {
    const Tensor &a = *this;

    if (a->device_ != b->device_)
        throw FatalExcept("tensor: operands for operator* are on different devices", __FILE__, __LINE__);
    if (a->dtype_ != b->dtype_)
        throw FatalExcept("tensor: operands for operator* have different data types", __FILE__, __LINE__);

    // [codegen] shape: ewise
    if (a->shape_ != b->shape_)
        throw FatalExcept(std::string() + "tensor: operator* cannot handle tensors " + std::string(a->shape_) + " and " + std::string(b->shape_), __FILE__ , __LINE__);

    switch (a->dtype_) {
    case ScalarType::fp32: {
        Tensor result(a->shape_, a->device_, a->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "mul_ewise(size, result, a, b)"
        dispatch_kernel(device).mul_ewise_fp32(result->shape_.size, result->data_, a->data_, b->data_);

        if ((a->requires_grad_ || b->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeMulFp32(result, a, b);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: operator* does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor Tensor::operator/(const Tensor &b) const {
    const Tensor &a = *this;

    if (a->device_ != b->device_)
        throw FatalExcept("tensor: operands for operator/ are on different devices", __FILE__, __LINE__);
    if (a->dtype_ != b->dtype_)
        throw FatalExcept("tensor: operands for operator/ have different data types", __FILE__, __LINE__);

    // [codegen] shape: ewise
    if (a->shape_ != b->shape_)
        throw FatalExcept(std::string() + "tensor: operator/ cannot handle tensors " + std::string(a->shape_) + " and " + std::string(b->shape_), __FILE__ , __LINE__);

    switch (a->dtype_) {
    case ScalarType::fp32: {
        Tensor result(a->shape_, a->device_, a->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "div_ewise(size, result, a, b)"
        dispatch_kernel(device).div_ewise_fp32(result->shape_.size, result->data_, a->data_, b->data_);

        if ((a->requires_grad_ || b->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeDivFp32(result, a, b);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: operator/ does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor square(const Tensor &t) {
    // [codegen] shape: identity

    switch (t->dtype_) {
    case ScalarType::fp32: {
        Tensor result(t->shape_, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "square(size, result, t)"
        dispatch_kernel(device).square_fp32(result->shape_.size, result->data_, t->data_);

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeSquareFp32(result, t);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: square does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor sqrt(const Tensor &t) {
    // [codegen] shape: identity

    switch (t->dtype_) {
    case ScalarType::fp32: {
        Tensor result(t->shape_, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "sqrt(size, result, t)"
        dispatch_kernel(device).sqrt_fp32(result->shape_.size, result->data_, t->data_);

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeSqrtFp32(result, t);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: sqrt does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor relu(const Tensor &t) {
    // [codegen] shape: identity

    switch (t->dtype_) {
    case ScalarType::fp32: {
        Tensor result(t->shape_, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "relu(size, result, t)"
        dispatch_kernel(device).relu_fp32(result->shape_.size, result->data_, t->data_);

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeReluFp32(result, t);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: relu does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor Tensor::operator+(float scalar) const {
    const Tensor &t = *this;

    // [codegen] shape: identity

    if (t->dtype_ != ScalarType::fp32)
        throw FatalExcept("tensor: operands for operator+ have different data types", __FILE__, __LINE__);
    else {
        Tensor result(t->shape_, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "add_scalar(size, result, t, scalar)"
        dispatch_kernel(device).add_scalar_fp32(result->shape_.size, result->data_, t->data_, scalar);

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeAddScalarFp32(result, t, scalar);
        }

        return result;
    }
}

inline Tensor Tensor::operator-(float scalar) const {
    const Tensor &t = *this;

    // [codegen] shape: identity

    if (t->dtype_ != ScalarType::fp32)
        throw FatalExcept("tensor: operands for operator- have different data types", __FILE__, __LINE__);
    else {
        Tensor result(t->shape_, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "add_scalar(size, result, t, -scalar)"
        dispatch_kernel(device).add_scalar_fp32(result->shape_.size, result->data_, t->data_, (-scalar));

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeSubScalarFp32(result, t, scalar);
        }

        return result;
    }
}

inline Tensor Tensor::operator*(float scalar) const {
    const Tensor &t = *this;

    // [codegen] shape: identity

    if (t->dtype_ != ScalarType::fp32)
        throw FatalExcept("tensor: operands for operator* have different data types", __FILE__, __LINE__);
    else {
        Tensor result(t->shape_, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "mul_scalar(size, result, t, scalar)"
        dispatch_kernel(device).mul_scalar_fp32(result->shape_.size, result->data_, t->data_, scalar);

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeMulScalarFp32(result, t, scalar);
        }

        return result;
    }
}

inline Tensor Tensor::operator/(float scalar) const {
    const Tensor &t = *this;

    // [codegen] shape: identity

    if (t->dtype_ != ScalarType::fp32)
        throw FatalExcept("tensor: operands for operator/ have different data types", __FILE__, __LINE__);
    else {
        Tensor result(t->shape_, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "mul_scalar(size, result, t, auto(1) / scalar)"
        dispatch_kernel(device).mul_scalar_fp32(result->shape_.size, result->data_, t->data_, 1.0f / scalar);

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeDivScalarFp32(result, t, scalar);
        }

        return result;
    }
}

inline Tensor pow(const Tensor &t, float scalar) {
    // [codegen] shape: identity

    if (t->dtype_ != ScalarType::fp32)
        throw FatalExcept("tensor: operands for pow have different data types", __FILE__, __LINE__);
    else {
        Tensor result(t->shape_, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "pow(size, result, t, scalar)"
        dispatch_kernel(device).pow_fp32(result->shape_.size, result->data_, t->data_, scalar);

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodePowFp32(result, t, scalar);
        }

        return result;
    }
}

inline Tensor add_broadcast(const Tensor &a, const Tensor &b) {
    if (a->device_ != b->device_)
        throw FatalExcept("tensor: operands for add_broadcast are on different devices", __FILE__, __LINE__);
    if (a->dtype_ != b->dtype_)
        throw FatalExcept("tensor: operands for add_broadcast have different data types", __FILE__, __LINE__);

    // [codegen] shape: broadcast
    size_t ndim = std::max({a->shape_.ndim, b->shape_.ndim});
    Workspace a_mask_workspace(ndim * sizeof(bool), DeviceType::cpu), b_mask_workspace(ndim * sizeof(bool), DeviceType::cpu), ret_dims_workspace(ndim * sizeof(size_t), DeviceType::cpu);
    bool *a_mask = a_mask_workspace, *b_mask = b_mask_workspace;
    size_t *ret_dims = ret_dims_workspace;
    size_t *a_dims = a->shape_.lengths.data(), *b_dims = b->shape_.lengths.data();
    for (size_t i = 0; i < ndim; ++i) {
        ret_dims[i] = 1;
        if (i < a->shape_.ndim) {
            if (a_dims[i] == 1)
                a_mask[i] = false;
            else {
                ret_dims[i] = a_dims[i];
                a_mask[i] = true;
            }
        }
        else
            a_mask[i] = false;
        if (i < b->shape_.ndim) {
            if (b_dims[i] == 1)
                b_mask[i] = false;
            else if (ret_dims[i] == 1) {
                ret_dims[i] = b_dims[i];
                b_mask[i] = true;
            }
            else if (ret_dims[i] == b_dims[i])
                b_mask[i] = true;
            else
                throw FatalExcept(std::string() + "tensor: add_broadcast cannot handle tensors " + std::string(a->shape_) + " and " + std::string(b->shape_), __FILE__ , __LINE__);
        }
        else
            b_mask[i] = false;
    }

    switch (a->dtype_) {
    case ScalarType::fp32: {
        Tensor result({ndim, ret_dims}, a->device_, a->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "add_broadcast(size, ndim, lengths, a.mask, b.mask, result, a, b)"
        dispatch_kernel(device).add_broadcast_fp32(result->shape_.size, result->shape_.ndim, result->shape_.lengths.data(), a_mask, b_mask, result->data_, a->data_, b->data_);

        if ((a->requires_grad_ || b->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeAddBroadcastFp32(result, a, b, std::move(a_mask_workspace), std::move(b_mask_workspace));
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: add_broadcast does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor sum(const Tensor &t, std::vector<size_t> dims) {
    // [codegen] shape: reduction
    size_t ndim = t->shape_.ndim;
    Workspace mask_workspace(ndim * sizeof(bool), DeviceType::cpu),
              ret_dims_workspace(ndim * sizeof(size_t), DeviceType::cpu);
    bool *mask = mask_workspace;
    size_t *ret_dims = ret_dims_workspace;
    for (size_t i = 0; i < ndim; i++)
        mask[i] = true;
    for (size_t &i: dims) {
        if (i < ndim)
            mask[i] = false;
    }
    for (size_t i = 0; i < ndim; i++)
        ret_dims[i] = mask[i] ? t->shape_.lengths[i] : 1;

    switch (t->dtype_) {
    case ScalarType::fp32: {
        Tensor result({ndim, ret_dims}, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "sum(t.size, ndim, t.lengths, $mask, result, t)"
        dispatch_kernel(device).sum_fp32(t->shape_.size, result->shape_.ndim, t->shape_.lengths.data(), mask, result->data_, t->data_);

        if ((t->requires_grad_) && !global_no_grad) {
            throw FatalExcept("tensor: sum does not have a corresponding autograd node", __FILE__, __LINE__);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: sum does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor cross_entropy(const Tensor &logits, const Tensor &label) {
    if (logits->device_ != label->device_)
        throw FatalExcept("tensor: operands for cross_entropy are on different devices", __FILE__, __LINE__);
    if (logits->dtype_ != label->dtype_)
        throw FatalExcept("tensor: operands for cross_entropy have different data types", __FILE__, __LINE__);

    // [codegen] shape: ewise
    if (logits->shape_ != label->shape_)
        throw FatalExcept(std::string() + "tensor: cross_entropy cannot handle tensors " + std::string(logits->shape_) + " and " + std::string(label->shape_), __FILE__ , __LINE__);

    switch (logits->dtype_) {
    case ScalarType::fp32: {
        Tensor result(logits->shape_, logits->device_, logits->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "log_softmax(size / lengths[0], lengths[0], result, logits)"
        dispatch_kernel(device).log_softmax_fp32(result->shape_.size / result->shape_.lengths[0], result->shape_.lengths[0], result->data_, logits->data_);

        // [codegen] "mul_scalar(size, result, result, auto(-1))"
        dispatch_kernel(device).mul_scalar_fp32(result->shape_.size, result->data_, result->data_, -1.0f);

        // [codegen] "mul_ewise(size, result, result, label)"
        dispatch_kernel(device).mul_ewise_fp32(result->shape_.size, result->data_, result->data_, label->data_);

        if ((logits->requires_grad_ || label->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeCrossEntropyFp32(result, logits, label);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: cross_entropy does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor maxpool(const Tensor &t, size_t h_stride, size_t w_stride) {
    // [codegen] shape: pooling
    size_t ndim = t->shape_.ndim;
    Workspace mask_workspace(t->shape_.size * sizeof(bool), t->device_),
              ret_dims_workspace(ndim * sizeof(size_t), DeviceType::cpu);
    bool *mask = mask_workspace;
    size_t *ret_dims = ret_dims_workspace, *src_dims = t->shape_.lengths.data();
    for (size_t i = 0; i < ndim; i++)
        ret_dims[i] = src_dims[i];
    ret_dims[1] = (ret_dims[1] - 1) / h_stride + 1;
    ret_dims[0] = (ret_dims[0] - 1) / w_stride + 1;

    switch (t->dtype_) {
    case ScalarType::fp32: {
        Tensor result({ndim, ret_dims}, t->device_, t->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "maxpool(t.size / t.lengths[1] / t.lengths[0], t.lengths[1], t.lengths[0], h_stride, w_stride, result, $mask, t)"
        dispatch_kernel(device).maxpool_fp32(t->shape_.size / t->shape_.lengths[1] / t->shape_.lengths[0], t->shape_.lengths[1], t->shape_.lengths[0], h_stride, w_stride, result->data_, mask, t->data_);

        if ((t->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeMaxpoolFp32(result, t, h_stride, w_stride, std::move(mask_workspace));
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: maxpool does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

template<bool transpose_a, bool transpose_b>
Tensor matmul(const Tensor &a, const Tensor &b) {
    if (a->device_ != b->device_)
        throw FatalExcept("tensor: operands for matmul are on different devices", __FILE__, __LINE__);
    if (a->dtype_ != b->dtype_)
        throw FatalExcept("tensor: operands for matmul have different data types", __FILE__, __LINE__);

    // [codegen] shape: matmul
    if (a->shape_.ndim != 2)
        throw FatalExcept(std::string() + "tensor: matmul cannot handle tensor a " + std::string(a->shape_)
                        + " with " + std::to_string(a->shape_.ndim) + " dimensions (2 expected)", __FILE__, __LINE__);
    if (b->shape_.ndim != 2)
        throw FatalExcept(std::string() + "tensor: matmul cannot handle tensor b " + std::string(b->shape_)
                        + " with " + std::to_string(b->shape_.ndim) + " dimensions (2 expected)", __FILE__, __LINE__);
    size_t matmul_m = transpose_a ? a->shape_.lengths[0] : a->shape_.lengths[1];
    size_t matmul_k = transpose_a ? a->shape_.lengths[1] : a->shape_.lengths[0];
    if (matmul_k != (transpose_b ? b->shape_.lengths[0] : b->shape_.lengths[1]))
        throw FatalExcept(std::string() + "tensor: incompatible shapes " + std::string(a->shape_) + (transpose_a ? "^T" : "")
                        + " and " + std::string(b->shape_) + (transpose_b ? "^T" : "") + " for matrix multiplication", __FILE__, __LINE__);
    size_t matmul_n = transpose_b ? b->shape_.lengths[1] : b->shape_.lengths[0];

    switch (a->dtype_) {
    case ScalarType::fp32: {
        Tensor result({matmul_m, matmul_n}, a->device_, a->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "gemm<a.transpose, b.transpose>($m, $k, $n, result, a, b)"
        dispatch_kernel(device).gemm_fp32[transpose_a][transpose_b](matmul_m, matmul_k, matmul_n, result->data_, a->data_, b->data_);

        if ((a->requires_grad_ || b->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeMatmulFp32(result, a, b, transpose_a, transpose_b, matmul_m, matmul_n, matmul_k);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: matmul does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}

inline Tensor conv(const Tensor &input, const Tensor &kernel, const Tensor &bias, size_t h_padding, size_t w_padding) {
    if (input->device_ != kernel->device_ || input->device_ != bias->device_)
        throw FatalExcept("tensor: operands for conv are on different devices", __FILE__, __LINE__);
    if (input->dtype_ != kernel->dtype_ || input->dtype_ != bias->dtype_)
        throw FatalExcept("tensor: operands for conv have different data types", __FILE__, __LINE__);

    // [codegen] shape: conv
    if (input->shape_.ndim != 4)
        throw FatalExcept(std::string() + "tensor: conv cannot handle tensor input " + std::string(input->shape_)
                        + " with " + std::to_string(input->shape_.ndim) + " dimensions (4 expected)", __FILE__, __LINE__);
    if (kernel->shape_.ndim != 4)
        throw FatalExcept(std::string() + "tensor: conv cannot handle tensor kernel " + std::string(kernel->shape_)
                        + " with " + std::to_string(kernel->shape_.ndim) + " dimensions (4 expected)", __FILE__, __LINE__);
    size_t conv_n = input->shape_.lengths[3],
           conv_ci = input->shape_.lengths[2],
           conv_co = kernel->shape_.lengths[3];
    if (kernel->shape_.lengths[2] != conv_ci)
        throw FatalExcept(std::string() + "tensor: convolution channel mismatch between input " + std::string(input->shape_)
                        + " and kernel " + std::string(kernel->shape_), __FILE__, __LINE__);
    if (bias->shape_.ndim != 4)
        throw FatalExcept(std::string() + "tensor: conv cannot handle tensor bias " + std::string(bias->shape_)
                        + " with " + std::to_string(bias->shape_.ndim) + " dimensions (4 expected)", __FILE__, __LINE__);
    if (bias->shape_.lengths[2] != conv_co || bias->shape_.size != bias->shape_.lengths[2])
        throw FatalExcept(std::string() + "tensor: incompatible bias shape " + std::string(bias->shape_) + " with output channels "
                        + std::to_string(conv_co) + ", expected [1, " + std::to_string(conv_co) + ", 1, 1]", __FILE__, __LINE__);
    if (h_padding >= kernel->shape_.lengths[1] || w_padding >= kernel->shape_.lengths[0])
        throw FatalExcept("tensor: padding cannot be larger than convolution kernel", __FILE__, __LINE__);
    size_t conv_h_out = input->shape_.lengths[1] - kernel->shape_.lengths[1] + 1 + h_padding * 2,
           conv_w_out = input->shape_.lengths[0] - kernel->shape_.lengths[0] + 1 + w_padding * 2;

    switch (input->dtype_) {
    case ScalarType::fp32: {
        Tensor result({conv_n, conv_co, conv_h_out, conv_w_out}, input->device_, input->dtype_);
        DeviceDesc device = result->device_;

        // [codegen] "conv($n, $ci, $co, input, $hi, $wi, kernel, $hk, $wk, h_padding, w_padding, bias, result)"
        dispatch_kernel(device).conv_fp32(conv_n, conv_ci, conv_co, input->data_, input->shape_.lengths[1], input->shape_.lengths[0], kernel->data_, kernel->shape_.lengths[1], kernel->shape_.lengths[0], h_padding, w_padding, bias->data_, result->data_);

        if ((input->requires_grad_ || kernel->requires_grad_ || bias->requires_grad_) && !global_no_grad) {
            result->requires_grad_ = true;
            result->grad_node_ = new GradNodeConvFp32(result, input, kernel, bias, h_padding, w_padding, conv_n, conv_ci, conv_co, conv_h_out, conv_w_out);
        }

        return result;
    }
    case ScalarType::int32:
        throw FatalExcept("tensor: conv does not support data type int32", __FILE__, __LINE__);
        break;
    default:
        throw FatalExcept("tensor: unknown data type", __FILE__, __LINE__);
        break;
    }
}