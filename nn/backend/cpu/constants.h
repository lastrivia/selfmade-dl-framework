#pragma once

#include <cstdint>

class cpu_constants {
public:
    static constexpr size_t CACHE_THRESHOLD = 1048576LL, THREAD_FLOPS_THRESHOLD = 1048576LL;
};

static constexpr size_t AVX2_FP32_N = 8;

static constexpr bool ENABLE_MULTITHREADING = true;