#include <iostream>
#include <vector>
#include <ranges>

#include "nn/nn.h"
#include "utils/progress_bar.h"

int main() {
    size_t batch_size = 64;
    std::vector<mnist_sample> train_dataset = mnist_loader::load(
        "../archive/train-images.idx3-ubyte",
        "../archive/train-labels.idx1-ubyte",
        batch_size
    );
    std::vector<mnist_sample> test_dataset = mnist_loader::load(
        "../archive/t10k-images.idx3-ubyte",
        "../archive/t10k-labels.idx1-ubyte",
        batch_size
    );
    fc_layer fc_0(784, 500), fc_1(500, 10);
    relu_layer relu;
    std::vector<nn_layer *> layers{&fc_0, &relu, &fc_1};

    adam_optimizer optimizer(0.001f);
    for (auto &layer: layers) {
        optimizer.register_layer(layer);
    }

    exponential_scheduler scheduler(0.8);
    scheduler.bind_optimizer(&optimizer);

    int train_loops = 50;

    for (int i = 0; i < train_loops; ++i) {
        std::cout << "epoch " << i + 1 << ':' << std::endl;
        progress_bar train_progress_bar(train_dataset.size(), 20, "[train]");
        train_progress_bar.start();
        for (auto &data: train_dataset) {
            tensor activation = data.data();
            for (auto layer: layers) {
                activation = layer->forward_propagation(activation);
            }
            tensor softmax_tensor = softmax(activation);
            tensor gradient = cross_entropy_grad(softmax_tensor, data.label());
            for (auto layer: std::ranges::reverse_view(layers)) {
                gradient = layer->back_propagation(gradient);
            }
            train_progress_bar.step();
            optimizer.step();
        }
        scheduler.step();
        progress_bar test_progress_bar(test_dataset.size(), 20, "[test] ");
        test_progress_bar.start();
        size_t correct = 0, total = 0;
        for (auto &data: test_dataset) {
            tensor activation = data.data();
            for (auto layer: layers) {
                activation = layer->forward_propagation(activation);
            }
            correct += data.correct_count(activation);
            total += data.batch_size();
            test_progress_bar.step();
        }
        std::cout << "correct: " << static_cast<double>(correct) / static_cast<double>(total) * 100.0 << "%" << std::endl;
    }
    return 0;
}
