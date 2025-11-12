#pragma once
        
#include "tensor.h"
#include "kernel_dispatcher.h"
#include "autograd_base.h"

class grad_node_copy_fp32 : public grad_node {
public:
    grad_node_copy_fp32(const tensor &result, const tensor &src) :
        grad_node(result), src_(src), src_ver_(src->version_) {}

    ~grad_node_copy_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (src_->requires_grad_)
            ret.push_back(src_.object_);
        return ret;
    }

    void backward() override {
        if (src_ver_ != src_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: src
        if (src_->requires_grad_) {

            // [codegen] "apply grad"
            dispatch_kernel(device).add_ewise_fp32(src_->shape_.size, src_->grad_data_, src_->grad_data_, tensor_->grad_data_);
        }
    }

private:
    tensor src_;
    size_t src_ver_;
};

class grad_node_copy_int32 : public grad_node {
public:
    grad_node_copy_int32(const tensor &result, const tensor &src) :
        grad_node(result), src_(src), src_ver_(src->version_) {}

    ~grad_node_copy_int32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (src_->requires_grad_)
            ret.push_back(src_.object_);
        return ret;
    }

    void backward() override {
        if (src_ver_ != src_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: src
        if (src_->requires_grad_) {

            // [codegen] "apply grad"
            dispatch_kernel(device).add_ewise_int32(src_->shape_.size, src_->grad_data_, src_->grad_data_, tensor_->grad_data_);
        }
    }

private:
    tensor src_;
    size_t src_ver_;
};

class grad_node_add_fp32 : public grad_node {
public:
    grad_node_add_fp32(const tensor &result, const tensor &a, const tensor &b) :
        grad_node(result), a_(a), b_(b), a_ver_(a->version_), b_ver_(b->version_) {}

    ~grad_node_add_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(2);
        if (a_->requires_grad_)
            ret.push_back(a_.object_);
        if (b_->requires_grad_)
            ret.push_back(b_.object_);
        return ret;
    }

    void backward() override {
        if (a_ver_ != a_->version_ || b_ver_ != b_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: a
        if (a_->requires_grad_) {

            // [codegen] "apply grad"
            dispatch_kernel(device).add_ewise_fp32(a_->shape_.size, a_->grad_data_, a_->grad_data_, tensor_->grad_data_);
        }

        // [codegen] backward: b
        if (b_->requires_grad_) {

            // [codegen] "apply grad"
            dispatch_kernel(device).add_ewise_fp32(b_->shape_.size, b_->grad_data_, b_->grad_data_, tensor_->grad_data_);
        }
    }

private:
    tensor a_, b_;
    size_t a_ver_, b_ver_;
};

class grad_node_sub_fp32 : public grad_node {
public:
    grad_node_sub_fp32(const tensor &result, const tensor &a, const tensor &b) :
        grad_node(result), a_(a), b_(b), a_ver_(a->version_), b_ver_(b->version_) {}

    ~grad_node_sub_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(2);
        if (a_->requires_grad_)
            ret.push_back(a_.object_);
        if (b_->requires_grad_)
            ret.push_back(b_.object_);
        return ret;
    }

    void backward() override {
        if (a_ver_ != a_->version_ || b_ver_ != b_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: a
        if (a_->requires_grad_) {

            // [codegen] "apply grad"
            dispatch_kernel(device).add_ewise_fp32(a_->shape_.size, a_->grad_data_, a_->grad_data_, tensor_->grad_data_);
        }

        // [codegen] backward: b
        if (b_->requires_grad_) {

            // [codegen] "workspace tmp b.size"
            workspace tmp(b_->shape_.size * sizeof(float), device);

            // [codegen] "mul_scalar(b.size, tmp, grad, auto(-1))"
            dispatch_kernel(device).mul_scalar_fp32(b_->shape_.size, tmp, tensor_->grad_data_, -1.0f);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(b_->shape_.size, b_->grad_data_, b_->grad_data_, tmp);
        }
    }

private:
    tensor a_, b_;
    size_t a_ver_, b_ver_;
};

class grad_node_mul_fp32 : public grad_node {
public:
    grad_node_mul_fp32(const tensor &result, const tensor &a, const tensor &b) :
        grad_node(result), a_(a), b_(b), a_ver_(a->version_), b_ver_(b->version_) {}

    ~grad_node_mul_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(2);
        if (a_->requires_grad_)
            ret.push_back(a_.object_);
        if (b_->requires_grad_)
            ret.push_back(b_.object_);
        return ret;
    }

    void backward() override {
        if (a_ver_ != a_->version_ || b_ver_ != b_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: a
        if (a_->requires_grad_) {

            // [codegen] "workspace tmp a.size"
            workspace tmp(a_->shape_.size * sizeof(float), device);

            // [codegen] "mul_ewise(a.size, tmp, grad, b)"
            dispatch_kernel(device).mul_ewise_fp32(a_->shape_.size, tmp, tensor_->grad_data_, b_->data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(a_->shape_.size, a_->grad_data_, a_->grad_data_, tmp);
        }

        // [codegen] backward: b
        if (b_->requires_grad_) {

            // [codegen] "workspace tmp b.size"
            workspace tmp(b_->shape_.size * sizeof(float), device);

            // [codegen] "mul_ewise(b.size, tmp, grad, a)"
            dispatch_kernel(device).mul_ewise_fp32(b_->shape_.size, tmp, tensor_->grad_data_, a_->data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(b_->shape_.size, b_->grad_data_, b_->grad_data_, tmp);
        }
    }

private:
    tensor a_, b_;
    size_t a_ver_, b_ver_;
};

class grad_node_div_fp32 : public grad_node {
public:
    grad_node_div_fp32(const tensor &result, const tensor &a, const tensor &b) :
        grad_node(result), a_(a), b_(b), a_ver_(a->version_), b_ver_(b->version_) {}

    ~grad_node_div_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(2);
        if (a_->requires_grad_)
            ret.push_back(a_.object_);
        if (b_->requires_grad_)
            ret.push_back(b_.object_);
        return ret;
    }

    void backward() override {
        if (a_ver_ != a_->version_ || b_ver_ != b_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: a
        if (a_->requires_grad_) {

            // [codegen] "workspace tmp a.size"
            workspace tmp(a_->shape_.size * sizeof(float), device);

            // [codegen] "div_ewise(a.size, tmp, grad, b)"
            dispatch_kernel(device).div_ewise_fp32(a_->shape_.size, tmp, tensor_->grad_data_, b_->data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(a_->shape_.size, a_->grad_data_, a_->grad_data_, tmp);
        }

        // [codegen] backward: b
        if (b_->requires_grad_) {

            // [codegen] "workspace tmp b.size"
            workspace tmp(b_->shape_.size * sizeof(float), device);

            // [codegen] "mul_scalar(b.size, tmp, grad, auto(-1))"
            dispatch_kernel(device).mul_scalar_fp32(b_->shape_.size, tmp, tensor_->grad_data_, -1.0f);

            // [codegen] "mul_ewise(b.size, tmp, tmp, a)"
            dispatch_kernel(device).mul_ewise_fp32(b_->shape_.size, tmp, tmp, a_->data_);

            // [codegen] "div_ewise(b.size, tmp, tmp, b)"
            dispatch_kernel(device).div_ewise_fp32(b_->shape_.size, tmp, tmp, b_->data_);

            // [codegen] "div_ewise(b.size, tmp, tmp, b)"
            dispatch_kernel(device).div_ewise_fp32(b_->shape_.size, tmp, tmp, b_->data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(b_->shape_.size, b_->grad_data_, b_->grad_data_, tmp);
        }
    }

private:
    tensor a_, b_;
    size_t a_ver_, b_ver_;
};

class grad_node_square_fp32 : public grad_node {
public:
    grad_node_square_fp32(const tensor &result, const tensor &t) :
        grad_node(result), t_(t), t_ver_(t->version_) {}

    ~grad_node_square_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "workspace tmp t.size"
            workspace tmp(t_->shape_.size * sizeof(float), device);

            // [codegen] "mul_ewise(t.size, tmp, grad, t)"
            dispatch_kernel(device).mul_ewise_fp32(t_->shape_.size, tmp, tensor_->grad_data_, t_->data_);

            // [codegen] "mul_scalar(t.size, tmp, tmp, auto(2))"
            dispatch_kernel(device).mul_scalar_fp32(t_->shape_.size, tmp, tmp, 2.0f);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tmp);
        }
    }

private:
    tensor t_;
    size_t t_ver_;
};

class grad_node_sqrt_fp32 : public grad_node {
public:
    grad_node_sqrt_fp32(const tensor &result, const tensor &t) :
        grad_node(result), t_(t), t_ver_(t->version_) {}

    ~grad_node_sqrt_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "workspace tmp t.size"
            workspace tmp(t_->shape_.size * sizeof(float), device);

            // [codegen] "sqrt(t.size, tmp, t)"
            dispatch_kernel(device).sqrt_fp32(t_->shape_.size, tmp, t_->data_);

            // [codegen] "mul_scalar(t.size, tmp, tmp, auto(2))"
            dispatch_kernel(device).mul_scalar_fp32(t_->shape_.size, tmp, tmp, 2.0f);

            // [codegen] "div_ewise(t.size, tmp, grad, tmp)"
            dispatch_kernel(device).div_ewise_fp32(t_->shape_.size, tmp, tensor_->grad_data_, tmp);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tmp);
        }
    }

private:
    tensor t_;
    size_t t_ver_;
};

class grad_node_relu_fp32 : public grad_node {
public:
    grad_node_relu_fp32(const tensor &result, const tensor &t) :
        grad_node(result), t_(t), t_ver_(t->version_) {}

    ~grad_node_relu_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "workspace tmp t.size"
            workspace tmp(t_->shape_.size * sizeof(float), device);

            // [codegen] "relu_backward(t.size, tmp, grad, t)"
            dispatch_kernel(device).relu_backward_fp32(t_->shape_.size, tmp, tensor_->grad_data_, t_->data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tmp);
        }
    }

private:
    tensor t_;
    size_t t_ver_;
};

class grad_node_add_scalar_fp32 : public grad_node {
public:
    grad_node_add_scalar_fp32(const tensor &result, const tensor &t, float scalar) :
        grad_node(result), t_(t), scalar_(scalar), t_ver_(t->version_) {}

    ~grad_node_add_scalar_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "apply grad"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tensor_->grad_data_);
        }
    }

private:
    tensor t_;
    float scalar_;
    size_t t_ver_;
};

class grad_node_sub_scalar_fp32 : public grad_node {
public:
    grad_node_sub_scalar_fp32(const tensor &result, const tensor &t, float scalar) :
        grad_node(result), t_(t), scalar_(scalar), t_ver_(t->version_) {}

    ~grad_node_sub_scalar_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "apply grad"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tensor_->grad_data_);
        }
    }

private:
    tensor t_;
    float scalar_;
    size_t t_ver_;
};

class grad_node_mul_scalar_fp32 : public grad_node {
public:
    grad_node_mul_scalar_fp32(const tensor &result, const tensor &t, float scalar) :
        grad_node(result), t_(t), scalar_(scalar), t_ver_(t->version_) {}

    ~grad_node_mul_scalar_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "workspace tmp t.size"
            workspace tmp(t_->shape_.size * sizeof(float), device);

            // [codegen] "mul_scalar(t.size, tmp, grad, scalar)"
            dispatch_kernel(device).mul_scalar_fp32(t_->shape_.size, tmp, tensor_->grad_data_, scalar_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tmp);
        }
    }

private:
    tensor t_;
    float scalar_;
    size_t t_ver_;
};

class grad_node_div_scalar_fp32 : public grad_node {
public:
    grad_node_div_scalar_fp32(const tensor &result, const tensor &t, float scalar) :
        grad_node(result), t_(t), scalar_(scalar), t_ver_(t->version_) {}

    ~grad_node_div_scalar_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "workspace tmp t.size"
            workspace tmp(t_->shape_.size * sizeof(float), device);

            // [codegen] "mul_scalar(t.size, tmp, grad, auto(1) / scalar)"
            dispatch_kernel(device).mul_scalar_fp32(t_->shape_.size, tmp, tensor_->grad_data_, 1.0f / scalar_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tmp);
        }
    }

private:
    tensor t_;
    float scalar_;
    size_t t_ver_;
};

class grad_node_pow_fp32 : public grad_node {
public:
    grad_node_pow_fp32(const tensor &result, const tensor &t, float scalar) :
        grad_node(result), t_(t), scalar_(scalar), t_ver_(t->version_) {}

    ~grad_node_pow_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "workspace tmp t.size"
            workspace tmp(t_->shape_.size * sizeof(float), device);

            // [codegen] "pow(t.size, tmp, t, scalar - auto(1))"
            dispatch_kernel(device).pow_fp32(t_->shape_.size, tmp, t_->data_, scalar_ - 1.0f);

            // [codegen] "mul_scalar(t.size, tmp, tmp, scalar)"
            dispatch_kernel(device).mul_scalar_fp32(t_->shape_.size, tmp, tmp, scalar_);

            // [codegen] "mul_ewise(t.size, tmp, tmp, grad)"
            dispatch_kernel(device).mul_ewise_fp32(t_->shape_.size, tmp, tmp, tensor_->grad_data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tmp);
        }
    }

private:
    tensor t_;
    float scalar_;
    size_t t_ver_;
};

class grad_node_add_broadcast_fp32 : public grad_node {
public:
    grad_node_add_broadcast_fp32(const tensor &result, const tensor &a, const tensor &b, workspace &&a_mask_workspace, workspace &&b_mask_workspace) :
        grad_node(result), a_(a), b_(b), a_mask_workspace_(std::move(a_mask_workspace)), b_mask_workspace_(std::move(b_mask_workspace)), a_ver_(a->version_), b_ver_(b->version_) {}

    ~grad_node_add_broadcast_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(2);
        if (a_->requires_grad_)
            ret.push_back(a_.object_);
        if (b_->requires_grad_)
            ret.push_back(b_.object_);
        return ret;
    }

    void backward() override {
        if (a_ver_ != a_->version_ || b_ver_ != b_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        bool *a_mask = a_mask_workspace_, *b_mask = b_mask_workspace_;

        device_desc device = tensor_->device();

        // [codegen] backward: a
        if (a_->requires_grad_) {
            if (a_->shape_.size == tensor_->shape_.size) {

                // [codegen] "apply grad"
                dispatch_kernel(device).add_ewise_fp32(a_->shape_.size, a_->grad_data_, a_->grad_data_, tensor_->grad_data_);
            }
            else {

                // [codegen] "workspace tmp a.size"
                workspace tmp(a_->shape_.size * sizeof(float), device);

                // [codegen] "sum(size, ndim, lengths, a.mask, tmp, grad)"
                dispatch_kernel(device).sum_fp32(tensor_->shape_.size, tensor_->shape_.ndim, tensor_->shape_.lengths.data(), a_mask, tmp, tensor_->grad_data_);

                // [codegen] "apply tmp"
                dispatch_kernel(device).add_ewise_fp32(a_->shape_.size, a_->grad_data_, a_->grad_data_, tmp);
            }
        }

        // [codegen] backward: b
        if (b_->requires_grad_) {
            if (b_->shape_.size == tensor_->shape_.size) {

                // [codegen] "apply grad"
                dispatch_kernel(device).add_ewise_fp32(b_->shape_.size, b_->grad_data_, b_->grad_data_, tensor_->grad_data_);
            }
            else {

                // [codegen] "workspace tmp b.size"
                workspace tmp(b_->shape_.size * sizeof(float), device);

                // [codegen] "sum(size, ndim, lengths, b.mask, tmp, grad)"
                dispatch_kernel(device).sum_fp32(tensor_->shape_.size, tensor_->shape_.ndim, tensor_->shape_.lengths.data(), b_mask, tmp, tensor_->grad_data_);

                // [codegen] "apply tmp"
                dispatch_kernel(device).add_ewise_fp32(b_->shape_.size, b_->grad_data_, b_->grad_data_, tmp);
            }
        }
    }

private:
    tensor a_, b_;
    workspace a_mask_workspace_, b_mask_workspace_;
    size_t a_ver_, b_ver_;
};

class grad_node_cross_entropy_fp32 : public grad_node {
public:
    grad_node_cross_entropy_fp32(const tensor &result, const tensor &logits, const tensor &label) :
        grad_node(result), logits_(logits), label_(label), logits_ver_(logits->version_), label_ver_(label->version_) {}

    ~grad_node_cross_entropy_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(2);
        if (logits_->requires_grad_)
            ret.push_back(logits_.object_);
        if (label_->requires_grad_)
            ret.push_back(label_.object_);
        return ret;
    }

    void backward() override {
        if (logits_ver_ != logits_->version_ || label_ver_ != label_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: logits
        if (logits_->requires_grad_) {

            // [codegen] "workspace tmp logits.size"
            workspace tmp(logits_->shape_.size * sizeof(float), device);

            // [codegen] "softmax(logits.size / logits.lengths[0], logits.lengths[0], tmp, logits)"
            dispatch_kernel(device).softmax_fp32(logits_->shape_.size / logits_->shape_.lengths[0], logits_->shape_.lengths[0], tmp, logits_->data_);

            // [codegen] "sub_ewise(logits.size, tmp, tmp, label)"
            dispatch_kernel(device).sub_ewise_fp32(logits_->shape_.size, tmp, tmp, label_->data_);

            // [codegen] "mul_ewise(logits.size, tmp, tmp, grad)"
            dispatch_kernel(device).mul_ewise_fp32(logits_->shape_.size, tmp, tmp, tensor_->grad_data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(logits_->shape_.size, logits_->grad_data_, logits_->grad_data_, tmp);
        }

        // [codegen] backward: label
        if (label_->requires_grad_) {

            // [codegen] "workspace tmp label.size"
            workspace tmp(label_->shape_.size * sizeof(float), device);

            // [codegen] "log_softmax(label.size / label.lengths[0], label.lengths[0], tmp, logits)"
            dispatch_kernel(device).log_softmax_fp32(label_->shape_.size / label_->shape_.lengths[0], label_->shape_.lengths[0], tmp, logits_->data_);

            // [codegen] "mul_scalar(label.size, tmp, tmp, auto(-1))"
            dispatch_kernel(device).mul_scalar_fp32(label_->shape_.size, tmp, tmp, -1.0f);

            // [codegen] "mul_ewise(label.size, tmp, tmp, grad)"
            dispatch_kernel(device).mul_ewise_fp32(label_->shape_.size, tmp, tmp, tensor_->grad_data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(label_->shape_.size, label_->grad_data_, label_->grad_data_, tmp);
        }
    }

private:
    tensor logits_, label_;
    size_t logits_ver_, label_ver_;
};

class grad_node_maxpool_fp32 : public grad_node {
public:
    grad_node_maxpool_fp32(const tensor &result, const tensor &t, size_t h_stride, size_t w_stride, workspace &&mask_workspace) :
        grad_node(result), t_(t), h_stride_(h_stride), w_stride_(w_stride), mask_workspace_(std::move(mask_workspace)), t_ver_(t->version_) {}

    ~grad_node_maxpool_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(1);
        if (t_->requires_grad_)
            ret.push_back(t_.object_);
        return ret;
    }

    void backward() override {
        if (t_ver_ != t_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        bool *mask = mask_workspace_;

        device_desc device = tensor_->device();

        // [codegen] backward: t
        if (t_->requires_grad_) {

            // [codegen] "workspace tmp t.size"
            workspace tmp(t_->shape_.size * sizeof(float), device);

            // [codegen] "maxpool_backward(t.size / t.lengths[1] / t.lengths[0], t.lengths[1], t.lengths[0], h_stride, w_stride, tmp, $mask, grad)"
            dispatch_kernel(device).maxpool_backward_fp32(t_->shape_.size / t_->shape_.lengths[1] / t_->shape_.lengths[0], t_->shape_.lengths[1], t_->shape_.lengths[0], h_stride_, w_stride_, tmp, mask, tensor_->grad_data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(t_->shape_.size, t_->grad_data_, t_->grad_data_, tmp);
        }
    }

private:
    tensor t_;
    size_t h_stride_, w_stride_;
    workspace mask_workspace_;
    size_t t_ver_;
};

class grad_node_matmul_fp32 : public grad_node {
public:
    grad_node_matmul_fp32(const tensor &result, const tensor &a, const tensor &b, bool transpose_a, bool transpose_b, size_t matmul_m, size_t matmul_n, size_t matmul_k) :
        grad_node(result), a_(a), b_(b), transpose_a_(transpose_a), transpose_b_(transpose_b), matmul_m_(matmul_m), matmul_n_(matmul_n), matmul_k_(matmul_k), a_ver_(a->version_), b_ver_(b->version_) {}

    ~grad_node_matmul_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(2);
        if (a_->requires_grad_)
            ret.push_back(a_.object_);
        if (b_->requires_grad_)
            ret.push_back(b_.object_);
        return ret;
    }

    void backward() override {
        if (a_ver_ != a_->version_ || b_ver_ != b_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: a
        if (a_->requires_grad_) {

            // [codegen] "workspace tmp a.size"
            workspace tmp(a_->shape_.size * sizeof(float), device);
            if (transpose_a_) {

                // [codegen] "gemm<b.transpose, true>($k, $n, $m, tmp, b, grad)"
                dispatch_kernel(device).gemm_fp32[transpose_b_][true](matmul_k_, matmul_n_, matmul_m_, tmp, b_->data_, tensor_->grad_data_);
            }
            else {

                // [codegen] "gemm<false, !b.transpose>($m, $n, $k, tmp, grad, b)"
                dispatch_kernel(device).gemm_fp32[false][(!transpose_b_)](matmul_m_, matmul_n_, matmul_k_, tmp, tensor_->grad_data_, b_->data_);
            }

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(a_->shape_.size, a_->grad_data_, a_->grad_data_, tmp);
        }

        // [codegen] backward: b
        if (b_->requires_grad_) {

            // [codegen] "workspace tmp b.size"
            workspace tmp(b_->shape_.size * sizeof(float), device);
            if (transpose_b_) {

                // [codegen] "gemm<true, a.transpose>($n, $m, $k, tmp, grad, a)"
                dispatch_kernel(device).gemm_fp32[true][transpose_a_](matmul_n_, matmul_m_, matmul_k_, tmp, tensor_->grad_data_, a_->data_);
            }
            else {

                // [codegen] "gemm<!a.transpose, false>($k, $m, $n, tmp, a, grad)"
                dispatch_kernel(device).gemm_fp32[(!transpose_a_)][false](matmul_k_, matmul_m_, matmul_n_, tmp, a_->data_, tensor_->grad_data_);
            }

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(b_->shape_.size, b_->grad_data_, b_->grad_data_, tmp);
        }
    }

private:
    tensor a_, b_;
    bool transpose_a_, transpose_b_;
    size_t matmul_m_, matmul_n_, matmul_k_;
    size_t a_ver_, b_ver_;
};

class grad_node_conv_fp32 : public grad_node {
public:
    grad_node_conv_fp32(const tensor &result, const tensor &input, const tensor &kernel, const tensor &bias, size_t h_padding, size_t w_padding, size_t conv_n, size_t conv_ci, size_t conv_co, size_t conv_h_out, size_t conv_w_out) :
        grad_node(result), input_(input), kernel_(kernel), bias_(bias), h_padding_(h_padding), w_padding_(w_padding), conv_n_(conv_n), conv_ci_(conv_ci), conv_co_(conv_co), conv_h_out_(conv_h_out), conv_w_out_(conv_w_out), input_ver_(input->version_), kernel_ver_(kernel->version_), bias_ver_(bias->version_) {}

    ~grad_node_conv_fp32() override = default;

    std::vector<tensor_impl *> inputs() override {
        std::vector<tensor_impl *> ret;
        ret.reserve(3);
        if (input_->requires_grad_)
            ret.push_back(input_.object_);
        if (kernel_->requires_grad_)
            ret.push_back(kernel_.object_);
        if (bias_->requires_grad_)
            ret.push_back(bias_.object_);
        return ret;
    }

    void backward() override {
        if (input_ver_ != input_->version_ || kernel_ver_ != kernel_->version_ || bias_ver_ != bias_->version_)
            throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
        device_desc device = tensor_->device();

        // [codegen] backward: input
        if (input_->requires_grad_) {

            // [codegen] "workspace tmp input.size"
            workspace tmp(input_->shape_.size * sizeof(float), device);

            // [codegen] "conv_input_grad($n, $ci, $co, grad, $ho, $wo, kernel, $hk, $wk, $hk - h_padding - 1, $wk - w_padding - 1, tmp)"
            dispatch_kernel(device).conv_input_grad_fp32(conv_n_, conv_ci_, conv_co_, tensor_->grad_data_, conv_h_out_, conv_w_out_, kernel_->data_, kernel_->shape_.lengths[1], kernel_->shape_.lengths[0], kernel_->shape_.lengths[1] - h_padding_ - 1, kernel_->shape_.lengths[0] - w_padding_ - 1, tmp);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(input_->shape_.size, input_->grad_data_, input_->grad_data_, tmp);
        }

        // [codegen] backward: kernel
        if (kernel_->requires_grad_) {

            // [codegen] "workspace tmp kernel.size"
            workspace tmp(kernel_->shape_.size * sizeof(float), device);

            // [codegen] "conv_kernel_grad($n, $ci, $co, input, $hi, $wi, grad, $ho, $wo, h_padding, w_padding, tmp)"
            dispatch_kernel(device).conv_kernel_grad_fp32(conv_n_, conv_ci_, conv_co_, input_->data_, input_->shape_.lengths[1], input_->shape_.lengths[0], tensor_->grad_data_, conv_h_out_, conv_w_out_, h_padding_, w_padding_, tmp);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(kernel_->shape_.size, kernel_->grad_data_, kernel_->grad_data_, tmp);
        }

        // [codegen] backward: bias
        if (bias_->requires_grad_) {

            // [codegen] "value bool sum_mask[4] = {false, false, true, false}"
            bool sum_mask[4] = {false, false, true, false};

            // [codegen] "workspace tmp bias.size"
            workspace tmp(bias_->shape_.size * sizeof(float), device);

            // [codegen] "sum(result.size, 4, result.lengths, sum_mask, tmp, grad)"
            dispatch_kernel(device).sum_fp32(tensor_->shape_.size, 4, tensor_->shape_.lengths.data(), sum_mask, tmp, tensor_->grad_data_);

            // [codegen] "apply tmp"
            dispatch_kernel(device).add_ewise_fp32(bias_->shape_.size, bias_->grad_data_, bias_->grad_data_, tmp);
        }
    }

private:
    tensor input_, kernel_, bias_;
    size_t h_padding_, w_padding_;
    size_t conv_n_, conv_ci_, conv_co_, conv_h_out_, conv_w_out_;
    size_t input_ver_, kernel_ver_, bias_ver_;
};