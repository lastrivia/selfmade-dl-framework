#pragma once

#include "tensor.h"
#include "kernel_dispatcher.h"
#include "autograd_base.h"
#include "autograd_engine.h"

// class grad_node_add_fp32 : public grad_node { // example
// public:
//     grad_node_add_fp32(const tensor &result, const tensor &a, const tensor &b) :
//         grad_node(result), a_(a), b_(b), a_ver_(a->version_), b_ver_(b->version_) {}
//
//     ~grad_node_add_fp32() override = default;
//
//     std::vector<tensor_impl *> inputs() override {
//         std::vector<tensor_impl *> ret;
//         ret.reserve(2);
//         if (a_->requires_grad_)
//             ret.push_back(a_.object_);
//         if (b_->requires_grad_)
//             ret.push_back(b_.object_);
//         return ret;
//     }
//
//     void backward() override {
//         if (a_ver_ != a_->version_ || b_ver_ != b_->version_)
//             throw nn_except("autograd: operands modified before backward", __FILE__, __LINE__);
//         device_desc device = tensor_->device();
//
//         // [codegen] backward: a
//         if (a_->requires_grad_) {
//
//             // [codegen] "return grad"
//             dispatch_kernel(device).add_ewise_fp32(a_->shape_.size, a_->grad_data_, a_->grad_data_, tensor_->grad_data_);
//         }
//
//         // [codegen] backward: b
//         if (b_->requires_grad_) {
//
//             // [codegen] "return grad"
//             dispatch_kernel(device).add_ewise_fp32(b_->shape_.size, b_->grad_data_, b_->grad_data_, tensor_->grad_data_);
//         }
//     }
//
// private:
//     tensor a_, b_;
//     size_t a_ver_, b_ver_;
// };

#include "autograd_impl.generated.h"
