#pragma once

#include <array>
#include <fstream>

#include "tensor.h"

class CS3339Data {
public:
    CS3339Data() : samples_(0), n_batches_(0) {}

    ~CS3339Data() {}

    void load(const std::string &filename, size_t batch_size = 0) {

        std::ifstream fin(filename, std::ios::binary);
        if (!fin.is_open()) {
            throw FatalExcept(std::string("dataset file ") + filename + " not found", __FILE__, __LINE__);
        }

        int32_t samples;
        fin.read(reinterpret_cast<char *>(&samples), sizeof(int32_t));
        samples_ = samples;

        if (batch_size == 0) {
            n_batches_ = 1;
            batch_size = samples;
        }
        else {
            n_batches_ = (samples + batch_size - 1) / batch_size;
        }

        data_.resize(n_batches_);
        label_.resize(n_batches_);
        batch_size_.resize(n_batches_);

        for (size_t i = 0; i < n_batches_ - 1; ++i)
            batch_size_[i] = batch_size;
        batch_size_[n_batches_ - 1] = samples - batch_size * (n_batches_ - 1);
        for (size_t i = 0; i < n_batches_; ++i) {
            data_[i] = Tensor({batch_size_[i], 512});
            label_[i] = Tensor({batch_size_[i], 100});
            label_[i].fill(0.0f);
        }

        constexpr size_t sample_bytes = 512 * sizeof(float) + sizeof(int32_t);
        std::array<uint8_t, sample_bytes> buf_space{};
        uint8_t *buf = buf_space.data();

        for (size_t i = 0; i < n_batches_; i++) {
            for (size_t j = 0; j < batch_size_[i]; j++) {
                fin.read(reinterpret_cast<char *>(buf), sample_bytes);
                for (size_t k = 0; k < 512; k++)
                    data_[i].at(j, k) = *(reinterpret_cast<float *>(buf) + k);
                int32_t label = *reinterpret_cast<int32_t *>(buf + 512 * sizeof(float));
                label_[i].at(j, label) = 1.0f;
            }
        }
    }

    const Tensor &data(size_t batch_idx = 0) const {
        return data_[batch_idx];
    }

    const Tensor &label(size_t batch_idx = 0) const {
        return label_[batch_idx];
    }

    size_t samples() const { return samples_; }

    size_t n_batches() const { return n_batches_; }

    void to_device(DeviceDesc device) {
        for (auto &x: data_)
            x.to_device(device);
        for (auto &x: label_)
            x.to_device(device);
    }

private:
    std::vector<Tensor> data_, label_;
    std::vector<size_t> batch_size_;
    size_t samples_, n_batches_;

};
