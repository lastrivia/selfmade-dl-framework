#pragma once

#include "base_layer.h"

class maxpool_layer : public nn_layer {
public:
    maxpool_layer(const size_t h_stride, const size_t w_stride)
        : h_stride_(h_stride), w_stride_(w_stride) {}

    ~maxpool_layer() override = default;

    tensor operator()(const tensor &x) override {
        return maxpool(x, h_stride_, w_stride_);
    }

    std::vector<tensor> enum_params() override {
        return {};
    }

    void to_device(device_desc device) override {}

private:
    size_t h_stride_, w_stride_;
};
