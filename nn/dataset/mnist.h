#pragma once

#include <cstdint>
#include <fstream>
#include <vector>

#include "../except.h"
#include "../backend.h"

class mnist_sample {
public:
    mnist_sample(size_t batch_size, const unsigned char *image_buf, const unsigned char *label_buf) :
        data_({batch_size, 1, 28, 28}),
        label_({batch_size, 10}),
        batch_size_(batch_size) {

        label_.fill(0.0f);
        for (size_t i = 0; i < batch_size_; ++i) {
            for (size_t j = 0; j < 28; ++j)
                for (size_t k = 0; k < 28; ++k)
                    data_.at(i, 0, j, k) = static_cast<float>(image_buf[i * 784 + j * 28 + k]) / 255.0f;
            size_t tag = label_buf[i];
            if (tag > 9)
                throw nn_except("invalid label from dataset file", __FILE__, __LINE__);
            label_.at(i, tag) = 1.0f;
        }
    }

    const tensor &data() const { return data_; }

    const tensor &label() const { return label_; }

    size_t batch_size() const { return batch_size_; }

    void to_device(device_desc device) {
        data_.to_device(device);
        label_.to_device(device);
    }

private:
    tensor data_;
    tensor label_;
    size_t batch_size_;
};

class mnist_loader {
    static uint32_t read_big_endian(std::istream &is) {
        unsigned char buf[4];
        is.read(reinterpret_cast<char *>(&buf), 4);
        uint32_t value = (static_cast<uint32_t>(buf[0]) << 24) |
                         (static_cast<uint32_t>(buf[1]) << 16) |
                         (static_cast<uint32_t>(buf[2]) << 8) |
                         (static_cast<uint32_t>(buf[3]));
        return value;
    }

public:
    static std::vector<mnist_sample> load(const std::string &image_file, const std::string &label_file,
                                          size_t batch_size) {
        std::vector<mnist_sample> result;
        std::ifstream image_is(image_file, std::ios::binary);
        std::ifstream label_is(label_file, std::ios::binary);
        uint32_t image_magic = read_big_endian(image_is),
                 image_count = read_big_endian(image_is),
                 image_rows = read_big_endian(image_is),
                 image_cols = read_big_endian(image_is);
        uint32_t label_magic = read_big_endian(label_is),
                 label_count = read_big_endian(label_is);
        if (image_magic != 2051 || label_magic != 2049 || image_count != label_count ||
            image_rows != 28 || image_cols != 28)
                throw nn_except("invalid mnist dataset", __FILE__, __LINE__);

        auto *image_buf = new unsigned char[784 * batch_size];
        auto *label_buf = new unsigned char[batch_size];
        for (size_t i = 0; i < image_count; i += batch_size) {
            size_t current_batch_size = std::min(image_count - i, batch_size);
            image_is.read(reinterpret_cast<char *>(image_buf), 784 * current_batch_size);
            label_is.read(reinterpret_cast<char *>(label_buf), current_batch_size);
            result.emplace_back(current_batch_size, image_buf, label_buf);
        }
        delete[] image_buf;
        delete[] label_buf;
        return result;
    }
};
