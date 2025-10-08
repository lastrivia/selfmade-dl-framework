#pragma once

#include "../except.h"
#include "tensor.h"
#include "kernel_dispatcher.h"
#include "utils.h"

inline void assert_data_type(const tensor &t, data_type dtype, const char *file, int line) {
    if (t.data_type_ != dtype)
        throw nn_except("tensor data type does not match", file, line);
}

inline void assert_type_consistency(const tensor &a, const tensor &b, const char *file, int line) {
    if (a.device_type_ != b.device_type_)
        throw nn_except("tensor devices do not match", file, line);
    if (a.data_type_ != b.data_type_)
        throw nn_except("tensor data types do not match", file, line);
}

inline void assert_shape_consistency(const tensor &a, const tensor &b, const char *file, int line) {
    if (a.samples_ != b.samples_ || a.channels_ != b.channels_ || a.height_ != b.height_ || a.width_ != b.width_)
        throw nn_except("tensor shapes do not match", file, line);
}

inline void assert_layout_consistency(const tensor &a, const tensor &b, const char *file, int line) {
    assert_type_consistency(a, b, file, line);
    assert_shape_consistency(a, b, file, line);
}

inline void assert_mask_consistency(const tensor &t, const tensor_mask &mask, const char *file, int line) {
    if (t.size() != mask.size_)
        throw nn_except("tensor mask size does not match", file, line);
    if (t.device_type_ != mask.device_type_)
        throw nn_except("tensor mask device does not match", file, line);
}

// ====== OPERATORS ======

template<bool transpose_a, bool transpose_b>
tensor matmul(const tensor &a, const tensor &b) {
    assert_type_consistency(a, b, __FILE__, __LINE__);
    if (a.samples_ != 1 || b.samples_ != 1 || a.channels_ != 1 || b.channels_ != 1)
        throw nn_except("matmul does not support batched data", __FILE__, __LINE__);
    if ((transpose_a ? a.height_ : a.width_) != (transpose_b ? b.width_ : b.height_))
        throw nn_except("matrix sizes do not match", __FILE__, __LINE__);

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
    assert_type_consistency(input, kernel, __FILE__, __LINE__);
    assert_type_consistency(input, bias, __FILE__, __LINE__);
    if (input.channels_ != kernel.channels_)
        throw nn_except("conv channels do not match", __FILE__, __LINE__);
    if (height_padding >= kernel.height_ || width_padding >= kernel.width_)
        throw nn_except("padding cannot be larger than kernel", __FILE__, __LINE__);

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
            input.data_, input.height_, input.width_,
            kernel.data_, kernel.height_, kernel.width_,
            height_padding, width_padding,
            bias.data_, ret.data_
        );
        break;
    }

    return ret;
}

inline tensor conv_input_grad(const tensor &output_grad, const tensor &kernel,
                              const size_t forward_height_padding, const size_t forward_width_padding) {
    // [n, c_o, h_o, w_o] * [c_o, c_i, h_k, w_k](rotated) -> [n, c_i, h_i, w_i]
    assert_type_consistency(output_grad, kernel, __FILE__, __LINE__);
    if (output_grad.channels_ != kernel.samples_)
        throw nn_except("conv channels do not match", __FILE__, __LINE__);
    if (forward_height_padding >= kernel.height_ || forward_width_padding >= kernel.width_)
        throw nn_except("padding cannot be larger than kernel", __FILE__, __LINE__);
    const size_t height_padding = kernel.height_ - forward_height_padding - 1,
                 width_padding = kernel.width_ - forward_width_padding - 1;

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
            output_grad.data_, output_grad.height_, output_grad.width_,
            kernel.data_, kernel.height_, kernel.width_,
            height_padding, width_padding,
            ret.data_
        );
        break;
    }

    return ret;
}

inline tensor conv_kernel_grad(const tensor &input, const tensor &output_grad,
                               const size_t height_padding, const size_t width_padding) {
    // [n, c_i, h_i, w_i] * [n, c_o, h_o, w_o] -> [c_o, c_i, h_k, w_k]
    assert_type_consistency(input, output_grad, __FILE__, __LINE__);
    if (input.samples_ != output_grad.samples_)
        throw nn_except("conv channels do not match", __FILE__, __LINE__);
    if (height_padding >= output_grad.height_ || width_padding >= output_grad.width_)
        throw nn_except("padding cannot be larger than kernel", __FILE__, __LINE__);

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
            input.data_, input.height_, input.width_,
            output_grad.data_, output_grad.height_, output_grad.width_,
            height_padding, width_padding,
            ret.data_
        );
        break;
    }

    return ret;
}

inline tensor &tensor::operator+=(const tensor &other) {
    assert_layout_consistency(other, *this, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).add_ewise_fp32(size(), data_, data_, other.data_);
        break;
    }
    return *this;
}

inline tensor operator+(const tensor &a, const tensor &b) {
    assert_layout_consistency(a, b, __FILE__, __LINE__);
    tensor ret(a.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).add_ewise_fp32(ret.size(), ret.data_, a.data_, b.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::operator-=(const tensor &other) {
    assert_layout_consistency(other, *this, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).sub_ewise_fp32(size(), data_, data_, other.data_);
        break;
    }
    return *this;
}

inline tensor operator-(const tensor &a, const tensor &b) {
    assert_layout_consistency(a, b, __FILE__, __LINE__);
    tensor ret(a.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sub_ewise_fp32(ret.size(), ret.data_, a.data_, b.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::mul_ewise(const tensor &other) {
    assert_layout_consistency(other, *this, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).mul_ewise_fp32(size(), data_, data_, other.data_);
        break;
    }
    return *this;
}

inline tensor mul_ewise(const tensor &a, const tensor &b) {
    assert_layout_consistency(a, b, __FILE__, __LINE__);
    tensor ret(a.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).mul_ewise_fp32(ret.size(), ret.data_, a.data_, b.data_);
        break;
    }
    return ret;
}

inline tensor &tensor::div_ewise(const tensor &other) {
    assert_layout_consistency(other, *this, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).div_ewise_fp32(size(), data_, data_, other.data_);
        break;
    }
    return *this;
}

inline tensor div_ewise(const tensor &a, const tensor &b) {
    assert_layout_consistency(a, b, __FILE__, __LINE__);
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

inline tensor &tensor::relu_backward(const tensor &input) {
    assert_layout_consistency(input, *this, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).relu_backward_fp32(size(), data_, data_, input.data_);
        break;
    }
    return *this;
}

inline tensor relu_backward(const tensor &t, const tensor &input) {
    assert_layout_consistency(t, input, __FILE__, __LINE__);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).relu_backward_fp32(ret.size(), ret.data_, t.data_, input.data_);
        break;
    }
    return ret;
}

// ====== FP32 OPERATORS ======

inline tensor &tensor::operator+=(float scalar) {
    assert_data_type(*this, data_type::fp32, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).add_scalar_fp32(size(), data_, data_, scalar);
        break;
    }
    return *this;
}

inline tensor operator+(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32, __FILE__, __LINE__);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).add_scalar_fp32(ret.size(), ret.data_, t.data_, scalar);
        break;
    }
    return ret;
}

inline tensor &tensor::operator-=(float scalar) {
    assert_data_type(*this, data_type::fp32, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).add_scalar_fp32(size(), data_, data_, -scalar);
        break;
    }
    return *this;
}

inline tensor operator-(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32, __FILE__, __LINE__);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).add_scalar_fp32(ret.size(), ret.data_, t.data_, -scalar);
        break;
    }
    return ret;
}

inline tensor &tensor::operator*=(float scalar) {
    assert_data_type(*this, data_type::fp32, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).mul_scalar_fp32(size(), data_, data_, scalar);
        break;
    }
    return *this;
}

inline tensor operator*(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32, __FILE__, __LINE__);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).mul_scalar_fp32(ret.size(), ret.data_, t.data_, scalar);
        break;
    }
    return ret;
}

inline tensor &tensor::operator/=(float scalar) {
    assert_data_type(*this, data_type::fp32, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).mul_scalar_fp32(size(), data_, data_, 1.0f / scalar);
        break;
    }
    return *this;
}

inline tensor operator/(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32, __FILE__, __LINE__);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).mul_scalar_fp32(ret.size(), ret.data_, t.data_, 1.0f / scalar);
        break;
    }
    return ret;
}

inline void broadcast(tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32, __FILE__, __LINE__);
    switch (t.data_type_) {
    case data_type::fp32:
        dispatch_kernel(t).broadcast_fp32(t.size(), t.data_, scalar);
        break;
    }
}

inline tensor &tensor::pow(float scalar) {
    assert_data_type(*this, data_type::fp32, __FILE__, __LINE__);
    switch (data_type_) {
    case data_type::fp32:
        dispatch_kernel(*this).pow_fp32(size(), data_, data_, scalar);
        break;
    }
    return *this;
}

inline tensor pow(const tensor &t, float scalar) {
    assert_data_type(t, data_type::fp32, __FILE__, __LINE__);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).pow_fp32(ret.size(), ret.data_, t.data_, scalar);
        break;
    }
    return ret;
}

// ====== TILE OPERATORS ======

inline tensor &tensor::add_tile(const tensor &tile) { // abandoned
    assert_type_consistency(*this, tile, __FILE__, __LINE__);
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
        throw nn_except("tensor tile shape does not match", __FILE__, __LINE__);
    return *this;
}

inline tensor add_tile(const tensor &t, const tensor &tile) { // abandoned
    assert_type_consistency(t, tile, __FILE__, __LINE__);
    tensor ret(t.height_, t.width_, t.device_type_, t.data_type_);
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
        throw nn_except("tensor tile shape does not match", __FILE__, __LINE__);
    return ret;
}

inline tensor &tensor::sub_tile(const tensor &tile) { // abandoned
    assert_type_consistency(*this, tile, __FILE__, __LINE__);
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
        throw nn_except("tensor tile shape does not match", __FILE__, __LINE__);
    return *this;
}

inline tensor sub_tile(const tensor &t, const tensor &tile) { // abandoned
    assert_type_consistency(t, tile, __FILE__, __LINE__);
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
        throw nn_except("tensor tile shape does not match", __FILE__, __LINE__);
    return ret;
}

inline tensor sum_rows(const tensor &t) { // abandoned
    tensor ret(1, t.width_, t.device_type_, t.data_type_);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sum_cyclic_fp32(t.height_, t.width_, ret.data_, t.data_);
        break;
    }
    return ret;
}

inline tensor sum_cols(const tensor &t) { // abandoned
    tensor ret(t.height_, 1, t.device_type_, t.data_type_);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sum_stretched_fp32(t.height_, t.width_, ret.data_, t.data_);
        break;
    }
    return ret;
}

inline tensor sum_by_channel(const tensor &t) { // abandoned
    tensor tmp(t.samples_, t.channels_, 1, 1, t.device_type_, t.data_type_);
    tensor ret(1, t.channels_, 1, 1, t.device_type_, t.data_type_);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sum_stretched_fp32(t.samples_ * t.channels_, t.height_ * t.width_, tmp.data_, t.data_);
        dispatch_kernel(ret).sum_cyclic_fp32(t.samples_, t.channels_, ret.data_, tmp.data_);
    }
    return ret;
}

inline tensor add_broadcast(const tensor &a, const tensor &b) {
    assert_type_consistency(a, b, __FILE__, __LINE__);
    size_t a_dim[4] = {a.width_, a.height_, a.channels_, a.samples_},
           b_dim[4] = {b.width_, b.height_, b.channels_, b.samples_},
           ret_dim[4];
    bool a_mask[4], b_mask[4];
    for (size_t i = 0; i < 4; i++) {
        if (a_dim[i] == b_dim[i]) {
            ret_dim[i] = a_dim[i];
            a_mask[i] = true;
            b_mask[i] = true;
        }
        else if (a_dim[i] == 1) {
            ret_dim[i] = b_dim[i];
            a_mask[i] = false;
            b_mask[i] = true;
        }
        else if (b_dim[i] == 1) {
            ret_dim[i] = a_dim[i];
            a_mask[i] = true;
            b_mask[i] = false;
        }
        else
            throw nn_except("tensor shapes does not match broadcast rules", __FILE__, __LINE__);
    }
    tensor ret(ret_dim[3], ret_dim[2], ret_dim[1], ret_dim[0], a.device_type_, a.data_type_);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).add_broadcast_fp32(ret.size(), 4, ret_dim, a_mask, b_mask, ret.data_, a.data_, b.data_);
        break;
    }
    return ret;
}

inline tensor sum(const tensor &t, std::vector<size_t> dims) {
    size_t t_dim[4] = {t.width_, t.height_, t.channels_, t.samples_},
    ret_dim[4] = {t.width_, t.height_, t.channels_, t.samples_};
    bool mask[4] = {true, true, true, true};
    for (size_t &i: dims) {
        mask[i] = false;
        ret_dim[i] = 1;
    }
    tensor ret(ret_dim[3], ret_dim[2], ret_dim[1], ret_dim[0], t.device_type_, t.data_type_);
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).sum_fp32(t.size(), 4, t_dim, mask, ret.data_, t.data_);
        break;
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

inline size_t correct_count(const tensor &out, const tensor &ans) {
    size_t ret;
    assert_layout_consistency(out, ans, __FILE__, __LINE__);
    switch (out.data_type_) {
    case data_type::fp32:
        dispatch_kernel(out).correct_count_fp32(out.height_, out.width_, &ret, out.data_, ans.data_);
        break;
    }
    return ret;
}

inline tensor maxpool(const tensor &t, tensor_mask &mask, const size_t h_stride, const size_t w_stride) {
    assert_mask_consistency(t, mask, __FILE__, __LINE__);
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
        throw nn_except("pooling tensor shape does not match", __FILE__, __LINE__);
    tensor ret(t.samples_, t.channels_, original_height, original_width, t.device_type_, t.data_type_);
    assert_mask_consistency(ret, mask, __FILE__, __LINE__);
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
