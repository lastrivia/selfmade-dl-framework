#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "../backend.h"

class mnist_sample {
public:
    mnist_sample(): data(784), label(0) {}

    tensor data;
    int label;

    tensor tag() const {
        tensor ret(10);
        for (int i = 0; i < 10; ++i)
            ret(i) = 0.0f;
        ret(label) = 1.0f;
        return ret;
    }

    int validate(const tensor &output) const {
        int predicted = 0;
        for (int i = 1; i < 10; ++i)
            if (output(i) > output(predicted))
                predicted = i;
        return (predicted == label) ? 1 : 0;
    }

    friend std::ostream &operator<<(std::ostream &os, const mnist_sample &sample) {
        os << "label: " << sample.label << std::endl;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                float x = sample.data(i * 28 + j);
                if (x > 0.67)
                    os << "##";
                else if (x > 0.34)
                    os << "++";
                else if (x > 0.01)
                    os << "--";
                else os << "  ";
            }
            os << std::endl;
        }
        return os;
    }
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
    static std::vector<mnist_sample> load(const std::string &image_file, const std::string &label_file) {
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
            throw std::invalid_argument("Invalid dataset");
        result.resize(image_count);
        for (int i = 0; i < image_count; ++i) {
            unsigned char buf[784];
            image_is.read(reinterpret_cast<char *>(buf), 784);
            for (int j = 0; j < 784; ++j) {
                result[i].data(j) = static_cast<float>(buf[j]) / 255.0f;
            }
        }
        for (int i = 0; i < image_count; ++i) {
            unsigned char buf;
            label_is.read(reinterpret_cast<char *>(&buf), 1);
            result[i].label = static_cast<int>(buf);
        }
        return result;
    }
};
