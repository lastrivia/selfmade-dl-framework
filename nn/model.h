#pragma once

#include <memory>
#include <ranges>
#include <vector>

#include "backend.h"
#include "layer/base_layer.h"

class nn_model {
public:
    template<typename... Layers>
    explicit nn_model(Layers &&... layers) {
        (add_layer(std::forward<Layers>(layers)), ...);
    }

    tensor forward_propagation(tensor activation) {
        for (auto &layer : layers_) {
            activation = layer->forward_propagation(activation);
        }
        return activation;
    }

    tensor back_propagation(tensor gradient) {
        for (auto &layer : std::ranges::reverse_view(layers_)) {
            gradient = layer->back_propagation(gradient);
        }
        return gradient;
    }

    void to_device(device_type_arg device) {
        for (auto &layer : layers_)
            layer->to_device(device);
    }

private:
    std::vector<std::unique_ptr<nn_layer> > layers_;

    template<typename layer_t>
        requires std::is_base_of_v<nn_layer, layer_t>
    void add_layer(layer_t l) {
        layers_.push_back(std::make_unique<layer_t>(std::move(l)));
    }

    friend class nn_optimizer;
};
