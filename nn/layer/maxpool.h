#pragma once

#include "base_layer.h"

class MaxpoolLayer : public Layer {
public:
    MaxpoolLayer(const size_t h_stride, const size_t w_stride)
        : h_stride_(h_stride), w_stride_(w_stride) {}

    ~MaxpoolLayer() override = default;

    Tensor operator()(const Tensor &x) override {
        return maxpool(x, h_stride_, w_stride_);
    }

    std::vector<Tensor> enum_params() override {
        return {};
    }

    void to_device(DeviceDesc device) override {}

private:
    size_t h_stride_, w_stride_;
};
