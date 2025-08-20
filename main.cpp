#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <ranges>
#include <vector>

class tensor;

class base_tensor {
public:
    virtual ~base_tensor() = default;

    virtual float &operator()(int, int) = 0;

    virtual const float &operator()(int, int) const = 0;

    virtual int rows() const = 0;

    virtual int cols() const = 0;

    virtual tensor operator+(const base_tensor &other) const;

    virtual tensor operator-(const base_tensor &other) const;

    virtual tensor operator*(const base_tensor &other) const;
};

class tensor : public base_tensor {
public:
    explicit tensor(const int rows): rows_(rows), cols_(1) {
        if (rows <= 0)
            throw std::invalid_argument("invalid size");
        data_ = new float[rows];
    }

    tensor(const int rows, const int cols): rows_(rows), cols_(cols) {
        if (rows <= 0 || cols <= 0)
            throw std::invalid_argument("invalid size");
        data_ = new float[rows * cols];
    }

    ~tensor() override {
        delete[] data_;
    }

    float &operator()(const int row, const int col) override {
        return data_[row * cols_ + col];
    }

    const float &operator()(const int row, const int col) const override {
        return data_[row * cols_ + col];
    }

    float &operator()(const int index) {
        return data_[index];
    }

    const float &operator()(const int index) const {
        return data_[index];
    }

    tensor(const tensor &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = new float[cols_ * rows_];
        memcpy(data_, other.data_, cols_ * rows_ * sizeof(float));
    }

    tensor(tensor &&other) noexcept {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        other.data_ = nullptr;
    }

    tensor &operator=(const tensor &other) {
        if (this == &other)
            return *this;
        rows_ = other.rows_;
        cols_ = other.cols_;
        delete[] data_;
        data_ = new float[cols_ * rows_];
        memcpy(data_, other.data_, cols_ * rows_ * sizeof(float));
        return *this;
    }

    tensor &operator=(tensor &&other) noexcept {
        rows_ = other.rows_;
        cols_ = other.cols_;
        delete[] data_;
        data_ = other.data_;
        other.data_ = nullptr;
        return *this;
    }

    class tensor_view : public base_tensor {
    public:
        friend class tensor;

        float &operator()(const int row, const int col) override {
            if (transposed_view_)
                return parent_.data_[col * parent_.cols_ + row];
            return parent_.data_[row * parent_.cols_ + col];
        }

        const float &operator()(const int row, const int col) const override {
            if (transposed_view_)
                return parent_.data_[col * parent_.cols_ + row];
            return parent_.data_[row * parent_.cols_ + col];
        }

        int rows() const override {
            return transposed_view_ ? parent_.cols_ : parent_.rows_;
        }

        int cols() const override {
            return transposed_view_ ? parent_.rows_ : parent_.cols_;
        }

        tensor operator+(const base_tensor &other) const override {
            return base_tensor::operator+(other);
        }

        tensor operator-(const base_tensor &other) const override {
            return base_tensor::operator-(other);
        }

        tensor operator*(const base_tensor &other) const override {
            return base_tensor::operator*(other);
        }

    private:
        const tensor &parent_;
        bool transposed_view_;

        explicit tensor_view(const tensor &parent, bool transposed_view): parent_(parent),
                                                                          transposed_view_(transposed_view) {}
    };

    explicit operator tensor_view() const {
        return tensor_view(*this, false);
    }

    tensor_view operator~() const {
        return tensor_view(*this, true);
    }

    tensor operator+(const base_tensor &other) const override {
        return base_tensor::operator+(other);
    }

    tensor operator-(const base_tensor &other) const override {
        return base_tensor::operator-(other);
    }

    tensor operator*(const base_tensor &other) const override {
        return base_tensor::operator*(other);
    }

    tensor &operator+=(const base_tensor &other) {
        if (rows_ != other.rows() || cols_ != other.cols())
            throw std::invalid_argument("Incompatible tensor sizes");
        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                (*this)(i, j) += other(i, j);
        return *this;
    }

    tensor &operator-=(const base_tensor &other) {
        if (rows_ != other.rows() || cols_ != other.cols())
            throw std::invalid_argument("Incompatible tensor sizes");
        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                (*this)(i, j) -= other(i, j);
        return *this;
    }

    tensor operator*(const float scalar) const {
        tensor result(rows_, cols_);
        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                result(i, j) = (*this)(i, j) * scalar;
        return result;
    }

    tensor &operator*=(const float scalar) {
        for (int i = 0; i < rows_ * cols_; ++i)
            data_[i] *= scalar;
        return *this;
    }

    int rows() const override {
        return rows_;
    }

    int cols() const override {
        return cols_;
    }

    friend std::istream &operator>>(std::istream &is, tensor &t) {
        for (int i = 0; i < t.rows_; ++i)
            for (int j = 0; j < t.cols_; ++j)
                is >> t(i, j);
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const tensor &t) {
        for (int i = 0; i < t.rows_; ++i) {
            for (int j = 0; j < t.cols_; ++j)
                os << t(i, j) << " ";
            os << std::endl;
        }
        return os;
    }

private:
    int rows_, cols_;
    float *data_;

    // tensor() = default;
};

tensor base_tensor::operator+(const base_tensor &other) const {
    if (rows() != other.rows() || cols() != other.cols())
        throw std::invalid_argument("Incompatible tensor sizes");
    tensor result(rows(), cols());
    for (int i = 0; i < rows(); i++)
        for (int j = 0; j < cols(); j++)
            result(i, j) = (*this)(i, j) + other(i, j);
    return result;
}

tensor base_tensor::operator-(const base_tensor &other) const {
    if (rows() != other.rows() || cols() != other.cols())
        throw std::invalid_argument("Incompatible tensor sizes");
    tensor result(rows(), cols());
    for (int i = 0; i < rows(); i++)
        for (int j = 0; j < cols(); j++)
            result(i, j) = (*this)(i, j) - other(i, j);
    return result;
}

tensor base_tensor::operator*(const base_tensor &other) const {
    if (cols() != other.rows())
        throw std::invalid_argument("Incompatible tensor sizes");
    tensor result(rows(), other.cols());
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < other.cols(); ++j) {
            result(i, j) = 0;
            for (int k = 0; k < cols(); ++k)
                result(i, j) += (*this)(i, k) * other(k, j);
        }
    }
    return result;
}

class nn_layer {
public:
    virtual ~nn_layer() = default;

    virtual tensor forward_propagation(const tensor &activation) = 0;

    virtual tensor back_propagation(const tensor &gradient, float learning_rate) = 0;
};

class fc_layer : public nn_layer {
public:
    fc_layer(const int input_size, const int output_size): weight_(output_size, input_size),
                                                           bias_(output_size),
                                                           input_(input_size) {}

    ~fc_layer() override = default;

    tensor forward_propagation(const tensor &input) override {
        input_ = input;
        return weight_ * input + bias_;
    }

    void random_init_he() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, sqrt(2.0 / static_cast<double>(input_.rows())));
        for (int i = 0; i < weight_.rows(); ++i)
            for (int j = 0; j < weight_.cols(); ++j)
                weight_(i, j) = static_cast<float>(dis(gen));
        for (int i = 0; i < bias_.rows(); ++i)
            bias_(i) = 0.0f;
    }

    tensor back_propagation(const tensor &output_grad, float learning_rate) override {
        const tensor weight_grad = output_grad * ~input_;
        const tensor &bias_grad = output_grad;
        weight_ -= weight_grad * learning_rate;
        bias_ -= bias_grad * learning_rate;
        tensor input_grad = ~weight_ * output_grad;
        return input_grad;
    }

private:
    tensor weight_, bias_, input_;
};

class relu_layer : public nn_layer {
public:
    explicit relu_layer() : input_(1) {}

    ~relu_layer() override = default;

    tensor forward_propagation(const tensor &input) override {
        input_ = input;
        tensor result = input;
        for (int i = 0; i < result.rows(); ++i) {
            if (result(i) < 0.0)
                result(i) = 0.0;
        }
        return result;
    }

    tensor back_propagation(const tensor &output_grad, float learning_rate) override {
        tensor input_grad = output_grad;
        for (int i = 0; i < input_grad.rows(); ++i) {
            if (input_(i) < 0.0)
                input_grad(i) = 0.0;
        }
        return input_grad;
    }

private:
    tensor input_;
};

tensor softmax(const tensor &input) {
    tensor result(input.rows());
    float sum = 0.0;
    for (int i = 0; i < result.rows(); ++i) {
        const float exp_result = exp(input(i));
        result(i) = exp_result;
        sum += exp_result;
    }
    for (int i = 0; i < result.rows(); ++i)
        result(i) /= sum;
    return result;
}

tensor cross_entropy_grad(const tensor &input_softmax, const tensor &tag) {
    return input_softmax - tag;
}

class progress_bar {
public:
    progress_bar(uint32_t steps, uint32_t length): steps_(steps), length_(length), current_step_(0), current_length_(0) {}

    void step() {
        ++current_step_;
        uint32_t new_length = current_step_ * length_ / steps_;
        if (new_length > current_length_) {
            current_length_ = new_length;
            std::cout << "=";
        }
        if (current_step_ == steps_)
            std::cout << std::endl;
    }

private:
    uint32_t steps_, length_, current_step_, current_length_;
};

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


int main() {
    std::vector<mnist_sample> train_dataset = mnist_loader::load(
        "../archive/train-images.idx3-ubyte",
        "../archive/train-labels.idx1-ubyte"
    );
    std::vector<mnist_sample> test_dataset = mnist_loader::load(
        "../archive/t10k-images.idx3-ubyte",
        "../archive/t10k-labels.idx1-ubyte"
    );
    fc_layer fc_0(784, 500), fc_1(500, 10);
    fc_0.random_init_he();
    fc_1.random_init_he();
    relu_layer relu;
    std::vector<nn_layer *> layers{&fc_0, &relu, &fc_1};

    int train_loops = 50;
    float learning_rate = 0.001f;

    for (int i = 0; i < train_loops; ++i) {
        std::cout << "train loop: " << i + 1 << std::endl;
        learning_rate *= 0.99;
        progress_bar train_progress(train_dataset.size(), 20);
        for (auto &data: train_dataset) {
            tensor activation = data.data;
            for (auto layer: layers) {
                activation = layer->forward_propagation(activation);
            }
            tensor softmax_tensor = softmax(activation);
            tensor gradient = cross_entropy_grad(softmax_tensor, data.tag());
            for (auto layer: std::ranges::reverse_view(layers)) {
                gradient = layer->back_propagation(gradient, learning_rate);
            }
            train_progress.step();
        }
        progress_bar test_progress(test_dataset.size(), 20);
        int correct = 0;
        for (auto &data: test_dataset) {
            tensor activation = data.data;
            for (auto layer: layers) {
                activation = layer->forward_propagation(activation);
            }
            correct += data.validate(activation);
            test_progress.step();
        }
        std::cout << "correct: " << static_cast<double>(correct) / static_cast<double>(test_dataset.size()) * 100.0 <<
                "%" << std::endl;
    }
    return 0;
}
