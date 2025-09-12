#include <iostream>
#include <vector>
#include <ranges>

#include "nn/nn.h"
#include "utils/progress_bar.h"

// #include "nn/backend/cpu/conv.h"

// int main() {
//     float in[] = {
//         1.0, 2.0, 3.0, 4.0, 5.0,
//         1.0, 2.0, 3.0, 4.0, 5.0,
//         1.0, 2.0, 3.0, 4.0, 5.0,
//         1.0, 2.0, 3.0, 4.0, 5.0,
//         1.0, 2.0, 3.0, 4.0, 5.0
//     };
//
//     float kernel[] = {
//         0.0, -1.0, 0.0,
//         0.0, 1.0, 0.0,
//         0.0, 0.0, 0.0,
//         0.0, 0.0, 0.0,
//         -1.0, 1.0, 0.0,
//         0.0, 0.0, 0.0
//     };
//
//     float out[50];
//
//     cpu_kernel::conv_fp32<true, true, false>(
//         1, 1, 2,
//         3, 3, 5, 5,
//         1, 1,
//         reinterpret_cast<char *>(out),
//         reinterpret_cast<char *>(in),
//         reinterpret_cast<char *>(kernel),
//         nullptr
//     );
//
//     for (int i = 0; i < 50; i += 5) {
//         for (int j = 0; j < 5; ++j)
//             std::cout << out[i + j] << " ";
//         std::cout << std::endl;
//     }
// }

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

    nn_model model(
        conv_layer(1, 4, 3, 3, 0, 0),
        flatten_layer(),
        fc_layer(2704, 256),
        relu_layer(true),
        fc_layer(256, 10)
    );

    adam_optimizer optimizer(0.001f);
    optimizer.register_model(model);

    exponential_scheduler scheduler(0.9);
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

            correct += data.correct_count(activation);
            total += data.batch_size();
            test_progress_bar.step();
        }
        std::cout << "correct: " << static_cast<double>(correct) / static_cast<double>(total) * 100.0 << "%" << std::endl;
    }
    return 0;
}
