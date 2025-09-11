#pragma once

#include <stdexcept>

#include "tensor.h"
#include "kernel_dispatcher.h"

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

// ====== OPERATORS ======

template<bool transpose_a, bool transpose_b>
tensor matmul(const tensor &a, const tensor &b) {
    assert_type_consistency(a, b);
    // if (a.samples_ != 1 || b.samples_ != 1 || a.channels_ != 1 || b.channels_ != 1)
    //     throw std::runtime_error("matmul does not support batched data");
    if ((transpose_a ? a.height_ : a.width_) != (transpose_b ? b.width_ : b.height_))
        throw std::runtime_error("matrix sizes does not match");

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
        dispatch_kernel(*this).relu_mask_fp32(size(), data_, data_, mask.data_);
        break;
    }
    return *this;
}

inline tensor relu_mask(const tensor &t, const tensor &mask) {
    assert_layout_consistency(t, mask);
    tensor ret(t.layout());
    switch (ret.data_type_) {
    case data_type::fp32:
        dispatch_kernel(ret).relu_mask_fp32(ret.size(), ret.data_, t.data_, mask.data_);
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
