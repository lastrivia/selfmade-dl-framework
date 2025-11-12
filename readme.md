Lightweight, self-made deep learning framework (i.e. a toy-class pytorch/tensorflow imitation), built on Windows

### Features
- Supported network architectures: MLP, CNN
- Runtime-selectable backends:
  - CPU (Manual SIMD & multithreading implementation)
  - CUDA
- Autograd system

### Build dependencies
- CUDA 12.0+, cuBLAS 12.0+, cuDNN 8.0+
- Python Interpreter 3.11+

### Roadmap

- Support for additional NN architectures (RNN, GAN, Transformer, etc.)
- Kernel/operator fusion via JIT compilation
- ONNX Model serialization and deserialization
- CPU-fallback compatible build

### v0.5 - 2025-11-12

- Autograd system
  - Runtime computation graph construction
- Reworked tensor type
  - Separated handle & implementation
  - Automatic garbage collection
  - Support for variable dimensions
- Code-generation-based build
  - Reduce code duplication and improve maintainability
- Universal broadcast & reduction operator

### 0924a (v0.4) - 2025-09-24

- CUDA backend
  - GEMM and convolution via cuBLAS/cuDNN 
  - Runtime backend switching support
- Refactors & improvements
  - Improved exception handling
  - More readable interface design

### 0915a (v0.3) - 2025-09-15

- Added CNN support
  - Additional layers: conv, maxpool, flatten
  - Multithreaded, SIMD conv kernel for improved performance 
- Refactors & improvements
  - Kernel dispatcher initialization & invocation
  - Model class for more convenient network operations
- Style improvement

### 0906a (v0.2.1) - 2025-09-06

- Optimized backend
  - Memory pool
  - Multithreading, SIMD, Strassen partition... for GEMM
- Misc fixes & style improvements

### 0905a (v0.2) - 2025-09-05

- Reworked backend framework
  - Separated kernel function implementation
  - Improved tensor class & interface
  - Kernel dispatcher to allow more devices in the future
  - Demo performance improvement ~30%
- Batched training

### 0826a (v0.1.1) - 2025-08-26

- Separated optimizer & scheduler class
- Adam optimizer

### 0820a (v0.1) - 2025-08-20

- Initial commit