// todo for memory alignment, unused yet

#pragma once

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#define assume_aligned(ptr, size) ptr = static_cast<decltype(ptr)>(__builtin_assume_aligned(ptr, size))

#elif defined(_MSC_VER)
#define assume_aligned(ptr, size) __assume((reinterpret_cast<std::uintptr_t>(ptr) % (size)) == 0)

#endif
