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

    for (auto &sample: train_dataset)
        sample.to_device(device_type::cuda);
    for (auto &sample: test_dataset)
        sample.to_device(device_type::cuda);


    // nn_model model(
    //     // [N, 1, 28, 28]
    //     conv_layer(1, 8, 3, 3, 1, 1),
    //     relu_layer(true),
    //     // [N, 8, 28, 28]
    //     conv_layer(8, 8, 3, 3, 1, 1),
    //     relu_layer(true),
    //     // [N, 8, 28, 28]
    //     maxpool_layer(2, 2),
    //     // [N, 8, 14, 14]
    //     conv_layer(8, 16, 3, 3, 0, 0),
    //     relu_layer(true),
    //     // [N, 16, 12, 12]
    //     conv_layer(16, 32, 3, 3, 0, 0),
    //     relu_layer(true),
    //     // [N, 32, 10, 10]
    //     maxpool_layer(2, 2),
    //     // [N, 32, 5, 5]
    //     flatten_layer(),
    //     // [1, 1, N, 800]
    //     fc_layer(800, 256),
    //     relu_layer(true),
    //     fc_layer(256, 10)
    // );

    nn_model model(
        // [N, 1, 28, 28]
        flatten_layer(),
        // [1, 1, N, 784]
        fc_layer(784, 256),
        relu_layer(true),
        fc_layer(256, 10)
    );
    model.to_device(device_type::cuda);

    adam_optimizer optimizer(0.001f);
    optimizer.register_model(model);

    exponential_scheduler scheduler(0.9f);
    scheduler.bind_optimizer(&optimizer);

    int train_loops = 50;

    for (int i = 0; i < train_loops; ++i) {
        std::cout << "epoch " << i + 1 << ':' << std::endl;

        progress_bar train_progress_bar(train_dataset.size(), 20, "[train]");
        train_progress_bar.start();
        for (auto &data: train_dataset) {

            tensor activation = model.forward_propagation(data.data());

            tensor softmax_tensor = softmax(activation);
            tensor gradient = cross_entropy_grad(softmax_tensor, data.label());

            model.back_propagation(gradient);

            optimizer.step();
            train_progress_bar.step();
        }
        scheduler.step();

        progress_bar test_progress_bar(test_dataset.size(), 20, "[test] ");
        test_progress_bar.start();
        size_t correct = 0, total = 0;
        for (auto &data: test_dataset) {

            tensor activation = model.forward_propagation(data.data());

            correct += correct_count(activation, data.label());
            total += data.batch_size();
            test_progress_bar.step();
        }
        std::cout << "correct: " << static_cast<double>(correct) / static_cast<double>(total) * 100.0 << "%" << std::endl;
    }
    return 0;
}
