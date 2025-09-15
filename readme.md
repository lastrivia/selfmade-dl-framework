A lightweight, self-made deep learning framework on Windows
- Manually implemented CPU backend
- Written in standard C++ with no third-party dependencies

### Currently in development

- CUDA backend
- Autograd mechanics
- Kernel operator fusion
- Additional layers (e.g., BatchNorm)

### 0915a - 2025-09-15

- Added CNN support
  - Additional layers: conv, maxpool, flatten
  - Multithreaded, SIMD conv kernel for improved performance 
- Refactorization and improvements
  - Kernel dispatcher initialization & invocation
  - Model class for more convenient network operations
- Style improvement

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