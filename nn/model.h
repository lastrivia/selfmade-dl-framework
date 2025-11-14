#pragma once

#include <memory>
#include <ranges>
#include <vector>

#include "tensor.h"
#include "layer/base_layer.h"

class Model {
public:
    template<typename... Layers>
    explicit Model(Layers &&... layers) {
        (add_layer(std::forward<Layers>(layers)), ...);
    }

    Tensor operator()(const Tensor &x) {
        Tensor activation = x;
        for (auto &layer : layers_) {
            activation = (*layer)(activation);
        }
        return activation;
    }

    void to_device(DeviceDesc device) {
        for (auto &layer : layers_)
            layer->to_device(device);
    }

private:
    std::vector<std::unique_ptr<Layer> > layers_;

    template<typename layer_t>
        requires std::is_base_of_v<Layer, layer_t>
    void add_layer(layer_t l) {
        layers_.push_back(std::make_unique<layer_t>(std::move(l)));
    }

    friend class Optimizer;
};
