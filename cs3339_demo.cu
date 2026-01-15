#include <iostream>
#include <ranges>

#include "nn/nn.h"
#include "nn/dataset/cs3339.h"
#include "nn/classical.h"
#include "utils/progress_bar.h"

int main() {

    size_t mlp_batch_size = 64;

    CS3339Data train_data, train_data_batched, val_data;
    train_data.load("../archive/train.dat");
    train_data_batched.load("../archive/train.dat", mlp_batch_size);
    val_data.load("../archive/val.dat");

    train_data.to_device("cuda");
    train_data_batched.to_device("cuda");
    val_data.to_device("cuda");

    // LR

    std::cout << "\nTesting LR model...\n" << std::endl;

    LogisticRegression lr_model(1.0, 0.001, 1000, 42, "cuda");
    lr_model.fit(train_data.data(), train_data.label());
    Tensor lr_logits = lr_model.predict(val_data.data());
    std::cout << "correct: " <<
        static_cast<double>(correct_count(lr_logits, val_data.label())) /
            static_cast<double>(val_data.samples()) * 100.0
    << "%" << std::endl;


    // NB

    std::cout << "\nTesting NB model...\n" << std::endl;

    GaussianNB nb_model("cuda");
    nb_model.fit(train_data.data(), train_data.label());
    Tensor nb_logits = nb_model.predict(val_data.data());
    std::cout << "correct: " <<
        static_cast<double>(correct_count(nb_logits, val_data.label())) /
            static_cast<double>(val_data.samples()) * 100.0
    << "%" << std::endl;


    // LDA

    // std::cout << "\nTesting LDA model...\n" << std::endl;
    //
    // LinearDiscriminantAnalysis lda_model("cuda");
    // lda_model.fit(train_data.data(), train_data.label());
    // Tensor lda_logits = lda_model.predict(val_data.data());
    // std::cout << "correct: " <<
    //     static_cast<double>(correct_count(lda_logits, val_data.label())) /
    //         static_cast<double>(val_data.samples()) * 100.0
    // << "%" << std::endl;


    // MLP

    std::cout << "\nTesting MLP model...\n" << std::endl;

    Model mlp_model(
        FCLayer(512, 256, 42),
        ReluLayer(),
        FCLayer(256, 256, 42),
        ReluLayer(),
        FCLayer(256, 100, 42)
    );
    mlp_model.to_device("cuda");

    AdamOptimizer optimizer(0.0001f);
    optimizer.register_model(mlp_model);

    int epochs = 10;

    for (int i = 0; i < epochs; ++i) {
        std::cout << "epoch " << i + 1 << ':' << std::endl;

        size_t n_batches = train_data_batched.n_batches();

        // ProgressBar train_progress_bar(n_batches, 20, "[train]");
        // train_progress_bar.start();
        for (size_t j = 0; j < n_batches; ++j) {

            Tensor logits = mlp_model(train_data_batched.data(j));
            Tensor loss = cross_entropy(logits, train_data_batched.label(j));

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // train_progress_bar.step();
        }

        size_t correct = 0, total = 0;

        {
            no_grad_lock lock;
            Tensor logits = mlp_model(val_data.data());

            correct += correct_count(logits, val_data.label());
            total += val_data.samples();
        }
        std::cout << "correct: " << static_cast<double>(correct) / static_cast<double>(total) * 100.0 << "%" << std::endl;
    }
    return 0;
}
