#pragma once

#include "immintrin.h"
#include <cstdint>
#include <thread>

namespace cpu_constants {
    static constexpr size_t CACHE_THRESHOLD = 1048576LL, THREAD_WORKLOAD_THRESHOLD = 1048576LL;

    static const size_t MAX_THREADS = std::thread::hardware_concurrency();
};

static constexpr size_t SSE_FP32_N = 4, AVX2_FP32_N = 8;

static constexpr bool ENABLE_MULTITHREADING = true;

inline float horizontal_sum_sse(const __m128 x) {
    __m128 hi2 = _mm_movehl_ps(x, x);
    __m128 sum2 = _mm_add_ps(x, hi2);
    __m128 hi = _mm_shuffle_ps(sum2, sum2, 0b00'00'00'01);
    __m128 sum = _mm_add_ss(sum2, hi);
    return _mm_cvtss_f32(sum);
}

inline float horizontal_sum_avx2(const __m256 x) {
    __m128 lo4 = _mm256_castps256_ps128(x);
    __m128 hi4 = _mm256_extractf128_ps(x, 1);
    __m128 sum4 = _mm_add_ps(lo4, hi4);
    __m128 hi2 = _mm_movehl_ps(sum4, sum4);
    __m128 sum2 = _mm_add_ps(sum4, hi2);
    __m128 hi = _mm_shuffle_ps(sum2, sum2, 0b00'00'00'01);
    __m128 sum = _mm_add_ss(sum2, hi);
    return _mm_cvtss_f32(sum);
}
