#include <iostream>
#include <vector>
#include <ranges>

#include "nn/nn.h"
#include "utils/progress_bar.h"

int main() {
    // tensor t(65, 8, 29, 29, device_type::cuda);
    // broadcast(t, 1.0f);
    // tensor s = sum(t, {0, 1, 3});
    // s.to_device("cpu");
    // std::cout << s.size() << std::endl;
    // for (int i = 0; i < 8; ++i)
    //     std::cout << s.at(i) << ' ';

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
        sample.to_device("cuda");
    for (auto &sample: test_dataset)
        sample.to_device("cuda");

    nn_model model(
        // [N, 1, 28, 28]
        conv_layer(1, 8, 3, 3, 1, 1),
        relu_layer(),
        // [N, 8, 28, 28]
        conv_layer(8, 8, 3, 3, 1, 1),
        relu_layer(),
        // [N, 8, 28, 28]
        maxpool_layer(2, 2),
        // [N, 8, 14, 14]
        conv_layer(8, 16, 3, 3, 0, 0),
        relu_layer(),
        // [N, 16, 12, 12]
        conv_layer(16, 32, 3, 3, 0, 0),
        relu_layer(),
        // [N, 32, 10, 10]
        maxpool_layer(2, 2),
        // [N, 32, 5, 5]
        flatten_layer(),
        // [1, 1, N, 800]
        fc_layer(800, 256),
        relu_layer(),
        fc_layer(256, 10)
    );

    std::cout << "\n == MODEL INIT FINISHED == \n" << std::endl;

    model.to_device("cuda");
    std::cout << "\n == MODEL TO DEVICE FINISHED == \n" << std::endl;

    adam_optimizer optimizer(0.001f);
    optimizer.register_model(model);
    std::cout << "\n == OPTIMIZER REGISTERED == \n" << std::endl;

    exponential_scheduler scheduler(0.9f);
    scheduler.bind_optimizer(&optimizer);

    std::cout << "\n == SCHEDULER BOUND == \n" << std::endl;

    int train_loops = 50;


    // mem_pool_log = true;
    // tensor_log = true;

    // cuda_mem_pool::query_allocated();

    for (int i = 0; i < train_loops; ++i) {
        std::cout << "epoch " << i + 1 << ':' << std::endl;

        progress_bar train_progress_bar(train_dataset.size(), 20, "[train]");
        train_progress_bar.start();
        for (auto &train_data: train_dataset) {


            // cuda_mem_pool::query_allocated();
            // mem_pool::query_allocated();

            // std::cout << "\n == FORWARD START == \n" << std::endl;

            tensor logits = model(train_data.data());
            tensor loss = cross_entropy(logits, train_data.label());


            // std::cout << "\n == BACKWARD START == \n" << std::endl;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            train_progress_bar.step();
        }
        scheduler.step();

        progress_bar test_progress_bar(test_dataset.size(), 20, "[test] ");
        test_progress_bar.start();
        size_t correct = 0, total = 0;
        for (auto &test_data: test_dataset) {

            // std::cout << "\n == EVAL FORWARD START == \n" << std::endl;

            tensor logits = model(test_data.data());

            correct += correct_count(logits, test_data.label());
            total += test_data.batch_size();
            test_progress_bar.step();
        }
        std::cout << "correct: " << static_cast<double>(correct) / static_cast<double>(total) * 100.0 << "%" << std::endl;
    }
    return 0;
}
