#include <fstream>
#include <iostream>
#include <ranges>
#include <vector>

#include "nn/nn.h"
#include "utils/progress_bar.h"

int main() {
    std::vector<mnist_sample> train_dataset = mnist_loader::load(
        "../archive/train-images.idx3-ubyte",
        "../archive/train-labels.idx1-ubyte"
    );
    std::vector<mnist_sample> test_dataset = mnist_loader::load(
        "../archive/t10k-images.idx3-ubyte",
        "../archive/t10k-labels.idx1-ubyte"
    );
    fc_layer fc_0(784, 200), fc_1(200, 10);
    relu_layer relu;
    std::vector<nn_layer *> layers{&fc_0, &relu, &fc_1};

    adam_optimizer optimizer(0.001f);
    for (auto &layer: layers) {
        optimizer.register_layer(layer);
    }

    exponential_scheduler scheduler(0.8);
    scheduler.bind_optimizer(&optimizer);

    int train_loops = 20;

    for (int i = 0; i < train_loops; ++i) {
        int start_time = clock();
        std::cout << "train loop: " << i + 1 << std::endl;
        progress_bar train_progress_bar(train_dataset.size(), 20);
        for (auto &data: train_dataset) {
            tensor activation = data.data;
            for (auto layer: layers) {
                activation = layer->forward_propagation(activation);
            }
            tensor softmax_tensor = softmax(activation);
            tensor gradient = cross_entropy_grad(softmax_tensor, data.tag());
            for (auto layer: std::ranges::reverse_view(layers)) {
                gradient = layer->back_propagation(gradient);
            }
            train_progress_bar.step();
            optimizer.step();
        }
        // std::string file_0 = std::string("param_a_") + std::to_string(i + 1) + ".csv";
        // std::string file_1 = std::string("param_b_") + std::to_string(i + 1) + ".csv";
        // fc_0.save_to(file_0);
        // fc_1.save_to(file_1);
        scheduler.step();
        progress_bar test_progress_bar(test_dataset.size(), 20);
        int correct = 0;
        for (auto &data: test_dataset) {
            tensor activation = data.data;
            for (auto layer: layers) {
                activation = layer->forward_propagation(activation);
            }
            correct += data.validate(activation);
            test_progress_bar.step();
        }
        std::cout << "correct: " << static_cast<double>(correct) / static_cast<double>(test_dataset.size()) * 100.0 <<
                "%" << std::endl;
        std::cout << "time elapsed: " << (clock() - start_time) / 1000.0 << "s" << std::endl;
    }
    return 0;
}
