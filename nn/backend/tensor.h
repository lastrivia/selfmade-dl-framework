#pragma once

#include "base_kernel.h"
#include "cpu/mem_pool.h"

class tensor {
public:
    struct layout_t {
        size_t samples, channels, height, width;
        device_type device_type;
        data_type data_type;

        bool operator==(const layout_t &layout) const = default;
    };

    tensor(
        size_t samples, size_t channels, size_t height, size_t width,
        device_type device_type = device_type::cpu,
        data_type data_type = data_type::fp32
    ) :
        samples_(samples), channels_(channels),
        height_(height), width_(width),
        device_type_(device_type), data_type_(data_type) {

        if (size() == 0) {
            data_ = nullptr;
            owns_data_ = false;
        }
        else {
            construct_data();
            owns_data_ = true;
        }
    }

    tensor(
        size_t height, size_t width,
        device_type device_type = device_type::cpu,
        data_type data_type = data_type::fp32
    ) :
        tensor(1, 1, height, width, device_type, data_type) {}

    explicit tensor(
        size_t size,
        device_type device_type = device_type::cpu,
        data_type data_type = data_type::fp32
    ) :
        tensor(1, 1, size, 1, device_type, data_type) {}

    tensor() : tensor(0, 0, 0, 0, device_type::cpu, data_type::fp32) {}

    explicit tensor(const layout_t &tensor_shape) : tensor(
        tensor_shape.samples, tensor_shape.channels, tensor_shape.height, tensor_shape.width,
        tensor_shape.device_type, tensor_shape.data_type
    ) {}

    [[nodiscard]] layout_t layout() const {
        return {samples_, channels_, height_, width_, device_type_, data_type_};
    }

    ~tensor() {
        if (owns_data_)
            release_data();
    }

    tensor &operator=(const tensor &other) {
        if (this != &other && data_ != other.data_) {
            if (other.owns_data_) { // deep copy
                if (owns_data_ && layout() == other.layout()) { // reuse data space
                    construct_data(false, other.data_);
                }
                else {
                    if (owns_data_)
                        release_data();
                    copy_layout(other);
                    construct_data(true, other.data_);
                }
                owns_data_ = true;
            }
            else { // other.owns_data_ == false; shallow copy
                if (owns_data_)
                    release_data();
                copy_layout(other);
                data_ = other.data_;
                owns_data_ = false;
            }
        }
        return *this;
    }

    tensor &operator=(tensor &&other) noexcept {
        if (this != &other && data_ != other.data_) {
            if (owns_data_)
                release_data();
            copy_layout(other);

            owns_data_ = other.owns_data_;
            data_ = other.data_;
            other.owns_data_ = false;
        }

        return *this;
    }

    tensor(const tensor &other) :
        samples_(other.samples_), channels_(other.channels_),
        height_(other.height_), width_(other.width_),
        device_type_(other.device_type_), data_type_(other.data_type_) {

        if (other.owns_data_) { // deep copy
            construct_data(true, other.data_);
            owns_data_ = true;
        }
        else {
            data_ = other.data_;
            owns_data_ = false;
        }
    }

    tensor(tensor &&other) noexcept :
        samples_(other.samples_), channels_(other.channels_),
        height_(other.height_), width_(other.width_),
        device_type_(other.device_type_), data_type_(other.data_type_) {

        owns_data_ = other.owns_data_;
        data_ = other.data_;
        other.owns_data_ = false;
    }

    template<typename T = float>
    T &at(size_t sample, size_t channel, size_t row_index, size_t col_index) {
        return *access_data<T>(((sample * channels_ + channel) * height_ + row_index) * width_ + col_index);
    }

    template<typename T = float>
    const T &at(size_t sample, size_t channel, size_t row_index, size_t col_index) const {
        return *access_data<T>(((sample * channels_ + channel) * height_ + row_index) * width_ + col_index);
    }

    template<typename T = float>
    T &at(size_t row_index, size_t col_index) {
        return *access_data<T>(row_index * width_ + col_index);
    }

    template<typename T = float>
    const T &at(size_t row_index, size_t col_index) const {
        return *access_data<T>(row_index * width_ + col_index);
    }

    template<typename T = float>
    T &at(size_t index) {
        return *access_data<T>(index);
    }

    template<typename T = float>
    const T &at(size_t index) const {
        return *access_data<T>(index);
    }

    [[nodiscard]] size_t size() const {
        return samples_ * channels_ * height_ * width_;
    }

    size_t samples() const { return samples_; }
    size_t channels() const { return channels_; }
    size_t height() const { return height_; }
    size_t width() const { return width_; }

    // ====== OPERATORS ======

    template<bool, bool>
    friend tensor matmul(const tensor &, const tensor &);

    friend tensor conv(const tensor &, const tensor &, const tensor &, size_t, size_t);
    friend tensor conv_input_grad(const tensor &, const tensor &, size_t, size_t);
    friend tensor conv_kernel_grad(const tensor &, const tensor &, size_t, size_t);

    friend tensor operator+(const tensor &, const tensor &);
    tensor &operator+=(const tensor &);
    friend tensor operator-(const tensor &, const tensor &);
    tensor &operator-=(const tensor &);
    friend tensor mul_ewise(const tensor &, const tensor &);
    tensor &mul_ewise(const tensor &);
    friend tensor div_ewise(const tensor &, const tensor &);
    tensor &div_ewise(const tensor &);

    friend tensor square(const tensor &);
    tensor &square();
    friend tensor sqrt(const tensor &);
    tensor &sqrt();
    friend tensor relu(const tensor &);
    tensor &relu();
    friend tensor relu_mask(const tensor &, const tensor &);
    tensor &relu_mask(const tensor &);

    friend tensor add_tile(const tensor &, const tensor &);
    tensor &add_tile(const tensor &);
    friend tensor sub_tile(const tensor &, const tensor &);
    tensor &sub_tile(const tensor &);
    friend tensor sum_rows(const tensor &);
    friend tensor sum_cols(const tensor &);
    friend tensor sum_by_channel(const tensor &);

    friend tensor softmax(const tensor &);
    tensor &softmax();

    friend class tensor_mask;
    friend tensor maxpool(const tensor &, tensor_mask &, size_t, size_t);
    friend tensor maxpool_backward(const tensor &, const tensor_mask &, size_t, size_t, size_t, size_t);

    // fp32
    friend tensor operator+(const tensor &, float);
    tensor &operator+=(float);
    friend tensor operator-(const tensor &, float);
    tensor &operator-=(float);
    friend tensor operator*(const tensor &, float);
    tensor &operator*=(float);
    friend tensor operator/(const tensor &, float);
    tensor &operator/=(float);

    friend void broadcast(tensor &, float);
    friend tensor pow(const tensor &, float);
    tensor &pow(float);

    // =======================

    friend const kernel &dispatch_kernel(const tensor &t);

    friend class flatten_layer;

protected:
    size_t samples_, channels_, height_, width_;

    // todo int device_id_;
    device_type device_type_;
    data_type data_type_;

    char *data_;
    bool owns_data_;

    friend void assert_data_type(const tensor &, data_type);
    friend void assert_type_consistency(const tensor &, const tensor &);
    friend void assert_shape_consistency(const tensor &, const tensor &);
    friend void assert_layout_consistency(const tensor &, const tensor &);
    friend void assert_mask_consistency(const tensor &, const tensor_mask &);

    void construct_data(bool alloc = true, const char *copy_src = nullptr) {
        switch (device_type_) {
        case device_type::cpu:
            switch (data_type_) {
            case data_type::fp32:
                if (alloc)
                    data_ = reinterpret_cast<char *>(mem_pool::alloc<float>(size()));
                if (copy_src)
                    memcpy(data_, copy_src, sizeof(float) * samples_ * channels_ * height_ * width_);
                break;
            }
            break;
        }
    }

    void release_data() {
        switch (device_type_) {
        case device_type::cpu:
            mem_pool::recycle(data_);
            break;
        default:
            break;
        }
        data_ = nullptr;
    }

    template<typename T = float>
    T *access_data(size_t offset) const {
        if constexpr (std::is_same_v<T, float>) {
            if (data_type_ == data_type::fp32)
                return reinterpret_cast<T *>(data_) + offset;
            throw std::runtime_error("accessed data type does not match");
        }
        else {
            throw std::runtime_error("unsupported data type");
        }
    }

    void copy_layout(const tensor &other) {
        samples_ = other.samples_;
        channels_ = other.channels_;
        height_ = other.height_;
        width_ = other.width_;
        device_type_ = other.device_type_;
        data_type_ = other.data_type_;
    }
};
