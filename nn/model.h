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

    tensor operator()(const tensor &x) {
        tensor activation = x;
        int layer_n = 0;
        for (auto &layer : layers_) {
            activation = (*layer)(activation);
            // cudaError_t err = cudaDeviceSynchronize();
            // if (err != cudaSuccess) {
            //     throw nn_except(std::string() + cudaGetErrorString(err) + " at layer " + std::to_string(layer_n), __FILE__, __LINE__);
            // }
            // else
            //     std::cout << "layer " << layer_n << " executed;" << std::endl;
            ++layer_n;
        }
        return activation;
    }

    void to_device(device_desc device) {
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
