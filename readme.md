A lightweight, self-made deep learning framework
- Manually implemented CPU backend
- Written in standard C++ with no third-party dependencies

### Currently in development

- CUDA backend
- Additional layers (e.g., conv2d)

### 0906a - 2025-09-06

- Optimized backend
  - Memory pool
  - Multithreading, SIMD, Strassen partition... for GEMM
- Misc fixes & style improvements

### 0905a - 2025-09-05

- A completely new backend framework
  - Separated kernel function implementation
  - Improved tensor class & interface
  - Kernel dispatcher to allow more devices in the future
  - Demo performance improvement ~30%
- Batched training

### 0826a - 2025-08-26

- Separated optimizer & scheduler class
- Adam optimizer

### 0820a - 2025-08-20

- Initial commit