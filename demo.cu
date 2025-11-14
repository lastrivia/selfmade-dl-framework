#include <iostream>
#include <vector>
#include <ranges>

#include "nn/nn.h"
#include "utils/progress_bar.h"

int main() {

    size_t batch_size = 64;
    std::vector<MnistSample> train_dataset = MnistLoader::load(
        "../archive/train-images.idx3-ubyte",
        "../archive/train-labels.idx1-ubyte",
        batch_size
    );
    std::vector<MnistSample> test_dataset = MnistLoader::load(
        "../archive/t10k-images.idx3-ubyte",
        "../archive/t10k-labels.idx1-ubyte",
        batch_size
    );

    for (auto &sample: train_dataset)
        sample.to_device("cuda");
    for (auto &sample: test_dataset)
        sample.to_device("cuda");

    Model model(
        // [N, 1, 28, 28]
        ConvLayer(1, 8, 3, 3, 1, 1),
        ReluLayer(),
        // [N, 8, 28, 28]
        ConvLayer(8, 8, 3, 3, 1, 1),
        ReluLayer(),
        // [N, 8, 28, 28]
        MaxpoolLayer(2, 2),
        // [N, 8, 14, 14]
        ConvLayer(8, 16, 3, 3, 0, 0),
        ReluLayer(),
        // [N, 16, 12, 12]
        ConvLayer(16, 32, 3, 3, 0, 0),
        ReluLayer(),
        // [N, 32, 10, 10]
        MaxpoolLayer(2, 2),
        // [N, 32, 5, 5]
        FlattenLayer(),
        // [1, 1, N, 800]
        FCLayer(800, 256),
        ReluLayer(),
        FCLayer(256, 10)
    );

    model.to_device("cuda");

    AdamOptimizer optimizer(0.001f);
    optimizer.register_model(model);

    ExponentialScheduler scheduler(0.9f);
    scheduler.bind_optimizer(&optimizer);

    int train_loops = 50;

    // mem_pool_log = true;
    // tensor_log = true;

    for (int i = 0; i < train_loops; ++i) {
        std::cout << "epoch " << i + 1 << ':' << std::endl;

        // cuda_mem_pool::query_allocated();
        // mem_pool::query_allocated();

        ProgressBar train_progress_bar(train_dataset.size(), 20, "[train]");
        train_progress_bar.start();
        for (auto &train_data: train_dataset) {

            Tensor logits = model(train_data.data());
            Tensor loss = cross_entropy(logits, train_data.label());

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            train_progress_bar.step();
        }
        scheduler.step();

        ProgressBar test_progress_bar(test_dataset.size(), 20, "[test] ");
        test_progress_bar.start();
        size_t correct = 0, total = 0;
        for (auto &test_data: test_dataset) {

            Tensor logits = model(test_data.data());

            correct += correct_count(logits, test_data.label());
            total += test_data.batch_size();
            test_progress_bar.step();
        }
        std::cout << "correct: " << static_cast<double>(correct) / static_cast<double>(total) * 100.0 << "%" << std::endl;
    }
    return 0;
}
