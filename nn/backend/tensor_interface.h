#pragma once

#include <stdexcept>

#include "tensor.h"
#include "kernel_dispatcher.h"
#include "utils.h"

inline void assert_data_type(const tensor &t, data_type dtype) {
    if (t.data_type_ != dtype)
        throw std::runtime_error("tensor data type does not match");
}

inline void assert_type_consistency(const tensor &a, const tensor &b) {
    if (a.device_type_ != b.device_type_)
        throw std::runtime_error("tensor devices do not match");
    if (a.data_type_ != b.data_type_)
        throw std::runtime_error("tensor data types do not match");
}

inline void assert_shape_consistency(const tensor &a, const tensor &b) {
    if (a.samples_ != b.samples_ || a.channels_ != b.channels_ || a.height_ != b.height_ || a.width_ != b.width_)
        throw std::runtime_error("tensor shapes do not match");
}

inline void assert_layout_consistency(const tensor &a, const tensor &b) {
    assert_type_consistency(a, b);
    assert_shape_consistency(a, b);
}

inline void assert_mask_consistency(const tensor &t, const tensor_mask &mask) {
    if (t.size() != mask.size_)
        throw std::runtime_error("tensor mask size does not match");
    if (t.device_type_ != mask.device_type_)
        throw std::runtime_error("tensor mask device does not match");
}

// ====== OPERATORS ======

template<bool transpose_a, bool transpose_b>
tensor matmul(const tensor &a, const tensor &b) {
    assert_type_consistency(a, b);
    // if (a.samples_ != 1 || b.samples_ != 1 || a.channels_ != 1 || b.channels_ != 1)
    //     throw std::runtime_error("matmul does not support batched data");
    if ((transpose_a ? a.height_ : a.width_) != (transpose_b ? b.width_ : b.height_))
        throw std::runtime_error("matrix sizes do not match");

    tensor ret(transpose_a ? a.width_ : a.height_, transpose_b ? b.height_ : b.width_, a.device_type_, a.data_type_);

    switch (a.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).gemm_fp32[transpose_a][transpose_b](
            transpose_a ? a.width_ : a.height_,
            transpose_a ? a.height_ : a.width_,
            transpose_b ? b.height_ : b.width_,
            ret.data_, a.data_, b.data_
        );
        break;
    }
    return ret;
}

inline tensor matmul(const tensor &a, const tensor &b) {
    return matmul<false, false>(a, b);
}

inline tensor conv(const tensor &input, const tensor &kernel, const tensor &bias,
                   const size_t height_padding, const size_t width_padding) {
    // [n, c_i, h_i, w_i] * [c_o, c_i, h_k, w_k] + bias -> [n, c_o, h_o, w_o]
    assert_type_consistency(input, kernel);
    assert_type_consistency(input, bias);
    if (input.channels_ != kernel.channels_)
        throw std::runtime_error("conv channels do not match");
    if (height_padding >= kernel.height_ || width_padding >= kernel.width_)
        throw std::runtime_error("padding cannot be larger than kernel");

    tensor ret(
        input.samples_, kernel.samples_,
        input.height_ - kernel.height_ + 1 + height_padding * 2,
        input.width_ - kernel.width_ + 1 + width_padding * 2,
        input.device_type_, input.data_type_
    );

    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).conv_fp32(
            input.samples_, input.channels_, kernel.samples_,
            input.height_, input.width_, kernel.height_, kernel.width_,
            height_padding, width_padding,
            ret.data_, input.data_, kernel.data_, bias.data_
        );
        break;
    }

    return ret;
}

inline tensor conv_input_grad(const tensor &output_grad, const tensor &kernel,
                              const size_t input_height_padding, const size_t input_width_padding) {
    // [n, c_o, h_o, w_o] * [c_o, c_i, h_k, w_k](rotated) -> [n, c_i, h_i, w_i]
    assert_type_consistency(output_grad, kernel);
    if (output_grad.channels_ != kernel.samples_)
        throw std::runtime_error("conv channels do not match");
    if (input_height_padding >= kernel.height_ || input_width_padding >= kernel.width_)
        throw std::runtime_error("padding cannot be larger than kernel");
    const size_t height_padding = kernel.height_ - input_height_padding - 1,
                 width_padding = kernel.width_ - input_width_padding - 1;

    tensor ret(
        output_grad.samples_, kernel.channels_,
        output_grad.height_ - kernel.height_ + 1 + height_padding * 2,
        output_grad.width_ - kernel.width_ + 1 + width_padding * 2,
        output_grad.device_type_, output_grad.data_type_
    );

    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).conv_input_grad_fp32(
            output_grad.samples_, kernel.channels_, output_grad.channels_,
            output_grad.height_, output_grad.width_, kernel.height_, kernel.width_,
            height_padding, width_padding,
            ret.data_, output_grad.data_, kernel.data_
        );
        break;
    }

    return ret;
}

inline tensor conv_kernel_grad(const tensor &input, const tensor &output_grad,
                               const size_t height_padding, const size_t width_padding) {
    // [n, c_i, h_i, w_i] * [n, c_o, h_o, w_o] -> [c_o, c_i, h_k, w_k]
    assert_type_consistency(input, output_grad);
    if (input.samples_ != output_grad.samples_)
        throw std::runtime_error("conv channels do not match");
    if (height_padding >= output_grad.height_ || width_padding >= output_grad.width_)
        throw std::runtime_error("padding cannot be larger than kernel");

    tensor ret(
        output_grad.channels_, input.channels_,
        input.height_ - output_grad.height_ + 1 + height_padding * 2,
        input.width_ - output_grad.width_ + 1 + width_padding * 2,
        input.device_type_, input.data_type_
    );

    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).conv_kernel_grad_fp32(
            input.samples_, input.channels_, output_grad.channels_,
            input.height_, input.width_, output_grad.height_, output_grad.width_,
            height_padding, width_padding,
            ret.data_, input.data_, output_grad.data_
        );
        break;
    }

    return ret;
}

inline tensor &tensor::operator+=(const tensor &other) {
    assert_layout_consistency(other, *this);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).add_ewise_fp32(size(), data_, data_, other.data_);
        break;
    }
    return *this;
}

inline tensor operator+(const tensor &a, const tensor &b) {
    assert_layout_consistency(a, b);
    tensor ret(a.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).add_ewise_fp32(ret.size(), ret.data_, a.data_, b.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::operator-=(const tensor &other) {
    assert_layout_consistency(other, *this);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).sub_ewise_fp32(size(), data_, data_, other.data_);
        break;
    }
    return *this;
}

inline tensor operator-(const tensor &a, const tensor &b) {
    assert_layout_consistency(a, b);
    tensor ret(a.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sub_ewise_fp32(ret.size(), ret.data_, a.data_, b.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::mul_ewise(const tensor &other) {
    assert_layout_consistency(other, *this);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).mul_ewise_fp32(size(), data_, data_, other.data_);
        break;
    }
    return *this;
}

inline tensor mul_ewise(const tensor &a, const tensor &b) {
    assert_layout_consistency(a, b);
    tensor ret(a.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).mul_ewise_fp32(ret.size(), ret.data_, a.data_, b.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::div_ewise(const tensor &other) {
    assert_layout_consistency(other, *this);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).div_ewise_fp32(size(), data_, data_, other.data_);
        break;
    }
    return *this;
}

inline tensor div_ewise(const tensor &a, const tensor &b) {
    assert_layout_consistency(a, b);
    tensor ret(a.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).div_ewise_fp32(ret.size(), ret.data_, a.data_, b.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::square() {
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).square_fp32(size(), data_, data_);
        break;
    }
    return *this;
}

inline tensor square(const tensor &t) {
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).square_fp32(ret.size(), ret.data_, t.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::sqrt() {
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).sqrt_fp32(size(), data_, data_);
        break;
    }
    return *this;
}

inline tensor sqrt(const tensor &t) {
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sqrt_fp32(ret.size(), ret.data_, t.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::relu() {
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).relu_fp32(size(), data_, data_);
        break;
    }
    return *this;
}

inline tensor relu(const tensor &t) {
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).relu_fp32(ret.size(), ret.data_, t.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::relu_mask(const tensor &mask) {
    assert_layout_consistency(mask, *this);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).relu_backward_fp32(size(), data_, data_, mask.data_);
        break;
    }
    return *this;
}

inline tensor relu_mask(const tensor &t, const tensor &mask) {
    assert_layout_consistency(t, mask);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).relu_backward_fp32(ret.size(), ret.data_, t.data_, mask.data_);
        break;
    }
    return ret;
}

// ====== FP32 OPERATORS ======

inline tensor &tensor::operator+=(float scalar) {
    assert_data_type(*this, data_type::fp32);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).add_scalar_fp32(size(), data_, data_, scalar);
        break;
    }
    return *this;
}

inline tensor operator+(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).add_scalar_fp32(ret.size(), ret.data_, t.data_, scalar);
        break;
    }
    return ret;
}

inline tensor &tensor::operator-=(float scalar) {
    assert_data_type(*this, data_type::fp32);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).add_scalar_fp32(size(), data_, data_, -scalar);
        break;
    }
    return *this;
}

inline tensor operator-(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).add_scalar_fp32(ret.size(), ret.data_, t.data_, -scalar);
        break;
    }
    return ret;
}

inline tensor &tensor::operator*=(float scalar) {
    assert_data_type(*this, data_type::fp32);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).mul_scalar_fp32(size(), data_, data_, scalar);
        break;
    }
    return *this;
}

inline tensor operator*(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).mul_scalar_fp32(ret.size(), ret.data_, t.data_, scalar);
        break;
    }
    return ret;
}

inline tensor &tensor::operator/=(float scalar) {
    assert_data_type(*this, data_type::fp32);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).mul_scalar_fp32(size(), data_, data_, 1.0f / scalar);
        break;
    }
    return *this;
}

inline tensor operator/(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).mul_scalar_fp32(ret.size(), ret.data_, t.data_, 1.0f / scalar);
        break;
    }
    return ret;
}

inline void broadcast(tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32);
    switch (t.data_type_) {
    case data_type::fp32:
        dispatch_kernel(t).broadcast_fp32(t.size(), t.data_, scalar);
        break;
    }
}

inline tensor &tensor::pow(float scalar) {
    assert_data_type(*this, data_type::fp32);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).pow_fp32(size(), data_, data_, scalar);
        break;
    }
    return *this;
}

inline tensor pow(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).pow_fp32(ret.size(), ret.data_, t.data_, scalar);
        break;
    }
    return ret;
}

// ====== TILE OPERATORS ======

inline tensor &tensor::add_tile(const tensor &tile) {
    assert_type_consistency(*this, tile);
    if (tile.height_ == height_ && tile.width_ == 1) {
        switch (data_type_) {
        case data_type::fp32:
            dispatch_kernel(*this).add_stretched_fp32(height_, width_, data_, data_, tile.data_);
            break;
        }
    }
    else if (tile.height_ == 1 && tile.width_ == width_) {
        switch (data_type_) {
        case data_type::fp32:
            dispatch_kernel(*this).add_cyclic_fp32(height_, width_, data_, data_, tile.data_);
            break;
        }
    }
    else
        throw std::invalid_argument("tensor tile shape does not match");
    return *this;
}

inline tensor add_tile(const tensor &t, const tensor &tile) {
    assert_type_consistency(t, tile);
    tensor ret(t.height_, t.width_);
    if (tile.height_ == t.height_ && tile.width_ == 1) {
        switch (ret.data_type_) {
        case data_type::fp32:
            dispatch_kernel(ret).add_stretched_fp32(ret.height_, ret.width_, ret.data_, t.data_, tile.data_);
            break;
        }
    }
    else if (tile.height_ == 1 && tile.width_ == t.width_) {
        switch (ret.data_type_) {
        case data_type::fp32:
            dispatch_kernel(ret).add_cyclic_fp32(ret.height_, ret.width_, ret.data_, t.data_, tile.data_);
            break;
        }
    }
    else
        throw std::invalid_argument("tensor tile shape does not match");
    return ret;
}

inline tensor &tensor::sub_tile(const tensor &tile) {
    assert_type_consistency(*this, tile);
    if (tile.height_ == height_ && tile.width_ == 1) {
        switch (data_type_) {
        case data_type::fp32:
            dispatch_kernel(*this).sub_stretched_fp32(height_, width_, data_, data_, tile.data_);
            break;
        }
    }
    else if (tile.height_ == 1 && tile.width_ == width_) {
        switch (data_type_) {
        case data_type::fp32:
            dispatch_kernel(*this).sub_cyclic_fp32(height_, width_, data_, data_, tile.data_);
            break;
        }
    }
    else
        throw std::invalid_argument("tensor tile shape does not match");
    return *this;
}

inline tensor sub_tile(const tensor &t, const tensor &tile) {
    assert_type_consistency(t, tile);
    tensor ret(t.height_, t.width_, t.device_type_, t.data_type_);
    if (tile.height_ == t.height_ && tile.width_ == 1) {
        switch (ret.data_type_) {
        case data_type::fp32:
            dispatch_kernel(ret).sub_stretched_fp32(ret.height_, ret.width_, ret.data_, t.data_, tile.data_);
            break;
        }
    }
    else if (tile.height_ == 1 && tile.width_ == t.width_) {
        switch (ret.data_type_) {
        case data_type::fp32:
            dispatch_kernel(ret).sub_cyclic_fp32(ret.height_, ret.width_, ret.data_, t.data_, tile.data_);
            break;
        }
    }
    else
        throw std::invalid_argument("tensor tile shape does not match");
    return ret;
}

inline tensor sum_rows(const tensor &t) {
    tensor ret(1, t.width_, t.device_type_, t.data_type_);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sum_cyclic_fp32(t.height_, t.width_, ret.data_, t.data_);
        break;
    }
    return ret;
}

inline tensor sum_cols(const tensor &t) {
    tensor ret(t.height_, 1, t.device_type_, t.data_type_);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sum_stretched_fp32(t.height_, t.width_, ret.data_, t.data_);
        break;
    }
    return ret;
}

inline tensor sum_by_channel(const tensor &t) {
    tensor tmp(t.samples_, t.channels_, 1, 1, t.device_type_, t.data_type_);
    tensor ret(1, t.channels_, 1, 1, t.device_type_, t.data_type_);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sum_stretched_fp32(t.samples_ * t.channels_, t.height_ * t.width_, tmp.data_, t.data_);
        dispatch_kernel(ret).sum_cyclic_fp32(t.samples_, t.channels_, ret.data_, tmp.data_);
    }
    return ret;
}

inline tensor &tensor::softmax() {
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).softmax_fp32(height_, width_, data_, data_);
        break;
    }
    return *this;
}

inline tensor softmax(const tensor &t) {
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).softmax_fp32(t.height_, t.width_, ret.data_, t.data_);
        break;
    }
    return ret;
}

inline tensor maxpool(const tensor &t, tensor_mask &mask, const size_t h_stride, const size_t w_stride) {
    assert_mask_consistency(t, mask);
    tensor ret(
        t.samples_, t.channels_,
        (t.height_ - 1) / h_stride + 1, (t.width_ - 1) / w_stride + 1,
        t.device_type_, t.data_type_
    );
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).maxpool_fp32(
            t.samples_ * t.channels_, t.height_, t.width_, h_stride, w_stride,
            ret.data_, mask.data_, t.data_
        );
        break;
    }
    return ret;
}

inline tensor maxpool_backward(const tensor &t, const tensor_mask &mask,
                               const size_t original_height, const size_t original_width,
                               const size_t h_stride, const size_t w_stride) {
    if (t.height_ != (original_height - 1) / h_stride + 1 || t.width_ != (original_width - 1) / w_stride + 1)
        throw std::invalid_argument("pooled tensor shape does not match");
    tensor ret(t.samples_, t.channels_, original_height, original_width, t.device_type_, t.data_type_);
    assert_mask_consistency(ret, mask);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).maxpool_backward_fp32(
            t.samples_ * t.channels_, original_height, original_width, h_stride, w_stride,
            ret.data_, mask.data_, t.data_
        );
        break;
    }
    return ret;
}
