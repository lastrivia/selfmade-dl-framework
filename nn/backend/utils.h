#pragma once

#include "tensor_interface.h"

inline tensor cross_entropy_grad(const tensor &input_softmax, const tensor &label) {
    return input_softmax - label;
}

class tensor_mask {

    tensor_mask() : size_(0), device_type_(device_type::cpu), data_(nullptr) {}

    void bind(const tensor &t) {
        if (t.size() == size_ && t.device_type_ == device_type_)
            return;

        switch (device_type_) {
        case device_type::cpu:
            mem_pool::recycle(data_);
        }

        size_ = t.size();
        device_type_ = t.device_type_;

        if (size_)
            switch (device_type_) {
            case device_type::cpu:
                data_ = mem_pool::alloc<char>(size_);
            }
    }

protected:
    size_t size_;
    device_type device_type_;
    char *data_;
};
