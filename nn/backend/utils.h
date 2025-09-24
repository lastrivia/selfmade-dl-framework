#pragma once

#include "tensor_interface.h"

inline tensor cross_entropy_grad(const tensor &input_softmax, const tensor &label) {
    return input_softmax - label;
}

class tensor_mask {
public:
    tensor_mask() : size_(0), device_type_(device_type::cpu), data_(nullptr) {}

    ~tensor_mask() {
        switch (device_type_) {
        case device_type::cpu:
            mem_pool::recycle(data_);
            break;
        case device_type::cuda:
            cuda_mem_pool::recycle(data_);
            break;
        default:
            break;
        }
    }

    void bind_layout(const tensor &t) {
        if (t.size() == size_ && t.device_type_ == device_type_)
            return;

        switch (device_type_) {
        case device_type::cpu:
            mem_pool::recycle(data_);
            break;
        case device_type::cuda:
            cuda_mem_pool::recycle(data_);
            break;
        default:
            throw nn_except("unsupported device type", __FILE__, __LINE__);
            break;
        }

        size_ = t.size();
        device_type_ = t.device_type_;

        if (size_)
            switch (device_type_) {
            case device_type::cpu:
                data_ = mem_pool::alloc<bool>(size_);
                break;
            case device_type::cuda:
                data_ = cuda_mem_pool::alloc<bool>(size_);
                break;
            default:
                throw nn_except("unsupported device type", __FILE__, __LINE__);
                break;
            }
    }

protected:
    size_t size_;
    device_type device_type_;
    bool *data_;

    friend tensor maxpool(const tensor &, tensor_mask &, size_t, size_t);
    friend tensor maxpool_backward(const tensor &, const tensor_mask &, size_t, size_t, size_t, size_t);

    friend void assert_mask_consistency(const tensor &, const tensor_mask &, const char*, int);
};
