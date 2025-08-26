#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <cstring>
#include <random>

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

    class transposed_view : public base_tensor {
    public:
        friend class tensor;

        float &operator()(const int row, const int col) override {
            return parent_.data_[col * parent_.cols_ + row];
        }

        const float &operator()(const int row, const int col) const override {
            return parent_.data_[col * parent_.cols_ + row];
        }

        int rows() const override {
            return parent_.cols_;
        }

        int cols() const override {
            return parent_.rows_;
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

        explicit transposed_view(const tensor &parent): parent_(parent) {}
    };

    transposed_view operator~() const {
        return transposed_view(*this);
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

inline tensor base_tensor::operator+(const base_tensor &other) const {
    if (rows() != other.rows() || cols() != other.cols())
        throw std::invalid_argument("Incompatible tensor sizes");
    tensor result(rows(), cols());
    for (int i = 0; i < rows(); i++)
        for (int j = 0; j < cols(); j++)
            result(i, j) = (*this)(i, j) + other(i, j);
    return result;
}

inline tensor base_tensor::operator-(const base_tensor &other) const {
    if (rows() != other.rows() || cols() != other.cols())
        throw std::invalid_argument("Incompatible tensor sizes");
    tensor result(rows(), cols());
    for (int i = 0; i < rows(); i++)
        for (int j = 0; j < cols(); j++)
            result(i, j) = (*this)(i, j) - other(i, j);
    return result;
}

inline tensor base_tensor::operator*(const base_tensor &other) const {
    if (cols() != other.rows())
        throw std::invalid_argument("Incompatible tensor sizes");
    tensor result(rows(), other.cols());
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < other.cols(); ++j) {
            result(i, j) = (*this)(i, 0) * other(0, j);
            for (int k = 1; k < cols(); ++k)
                result(i, j) += (*this)(i, k) * other(k, j);
        }
    }
    return result;
}

inline tensor softmax(const tensor &input) {
    tensor result(input.rows());
    float max_val = input(0);
    for (int i = 1; i < input.rows(); ++i) {
        if (input(i) > max_val)
            max_val = input(i);
    }
    float sum = 0.0;
    for (int i = 0; i < input.rows(); ++i) {
        float exp_result = std::exp(input(i) - max_val);
        result(i) = exp_result;
        sum += exp_result;
    }
    for (int i = 0; i < input.rows(); ++i)
        result(i) /= sum;
    return result;
}

inline tensor per_element_sqr(const tensor &input) {
    tensor result(input.rows(), input.cols());
    for (int i = 0; i < result.rows(); ++i)
        for (int j = 0; j < result.cols(); ++j)
            result(i, j) = input(i, j) * input(i, j);
    return result;
}

inline tensor cross_entropy_grad(const tensor &input_softmax, const tensor &tag) {
    return input_softmax - tag;
}
