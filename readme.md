A lightweight, self-made deep learning framework on Windows

- CPU Backend
  - Manual C++ implementation with no third-party dependencies
  - SIMD acceleration & multithreading support
- CUDA Backend
  - Dependencies: CUDA (12.0+), cuBLAS (12.0+), cuDNN (8.0+)

### Currently in development

- Extend support for more NN architectures (RNN, GAN, Transformer, etc.)
- Autograd system
- Kernel operator fusion (JIT compilation)
- Manual CUDA backend implementation (without cuBLAS/cuDNN)
- Model serialization and deserialization
- CPU-fallback compatible build

### 0924a - 2025-09-24

- CUDA backend
  - GEMM and convolution via cuBLAS/cuDNN 
  - Runtime backend switching support
- Refactors & improvements
  - Improved exception handling
  - More readable interface design

### 0915a - 2025-09-15

- Added CNN support
  - Additional layers: conv, maxpool, flatten
  - Multithreaded, SIMD conv kernel for improved performance 
- Refactors & improvements
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