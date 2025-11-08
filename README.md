# CUDA Matrix Multiplication Optimization

> **Note:** This repository contains a **technical lab report** for a university course on GPU programming.  
> The goal is to explore CUDA optimization strategies for matrix multiplication.

---

## 🧩 Overview

This project benchmarks three CUDA implementations of single-precision matrix multiplication (SGEMM) on an **NVIDIA GeForce RTX 4060 GPU**.  
By introducing **warp-aligned memory access** and **shared memory tiling**, we analyze how each strategy impacts computational throughput and bandwidth utilization.

---

## ⚙️ Hardware Specification

| Component            | Detail                         |
| -------------------- | ------------------------------ |
| **GPU**              | NVIDIA GeForce RTX 4060        |
| **Memory**           | 8 GB GDDR6, 272 GB/s bandwidth |
| **Clock Speed**      | 1830 – 2460 MHz                |
| **CUDA Cores**       | 3072                           |
| **Theoretical Peak** | 15.11 TFLOPS ≈ 15 110 GFLOPS   |

Reference: [TechPowerUp GPU Specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-4060.c4107)

---

## 🧠 Experiment Setup

| Parameter          | Value                                |
| ------------------ | ------------------------------------ |
| Matrix Size        | N = 1024                             |
| Thread Block       | 32 × 32                              |
| Scalar Multipliers | `a` and `b` in SGEMM formula         |
| Kernel Variants    | Naive / Warp-Aligned / Shared-Memory |

---

## 🧮 Implementations

### 1. Naive Kernel
Baseline kernel directly computes `C = a * A × B + b * C` using global memory only.

```cpp
__global__ void sgemm(float *A, float *B, float *C, int n, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float tmp = 0;
        for (int k = 0; k < n; k++)
            tmp += A[i * n + k] * B[k * n + j];
        C[i * n + j] = a * tmp + b * C[i * n + j];
    }
}
```

------

### 2. Warp-Aligned Kernel

Reorders indexing to align memory access with 32-thread warps, improving coalesced reads/writes.

```cpp
#define BLOCK_SIZE 32
__global__ void sgemm(float *A, float *B, float *C, int n, float a, float b) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float tmp = 0;
        for (int k = 0; k < n; k++)
            tmp += A[i * n + k] * B[k * n + j];
        C[i * n + j] = a * tmp + b * C[i * n + j];
    }
}
```

------

### 3. Shared-Memory Tiling

Introduces shared memory tiles (`As`, `Bs`) to minimize global memory traffic and reuse sub-blocks.

```cpp
__global__ void sgemm(float *A, float *B, float *C, int n, float a, float b) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStep = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * n;

    float Csub = 0.0f;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    int cIndex = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[cIndex + n * ty + tx] = a * Csub + b * C[cIndex + n * ty + tx];
}
```

------

## 📊 Results

| Strategy          | GFLOPS/s | Bandwidth (GB/s) | Time (s) | % of Peak FLOPS | % of Peak Bandwidth |
| ----------------- | -------- | ---------------- | -------- | --------------- | ------------------- |
| **Naive**         | 86.19    | 0.17             | 0.0249   | 0.57 %          | 0.06 %              |
| **Warp-Aligned**  | 710.88   | 1.39             | 0.0030   | 4.70 %          | 0.51 %              |
| **Shared Memory** | 926.86   | 1.81             | 0.0023   | 6.13 %          | 0.67 %              |

Even with optimizations, performance remains well below the theoretical GPU peak, largely due to:

- High synchronization cost (`__syncthreads()` twice per tile)
- Shared memory bank conflicts
- Single-precision only; half-precision (FP16) could improve throughput
- Potential for loop unrolling and register tiling

------

## 🧾 Takeaways

- Memory access patterns and data reuse dominate GPU performance.
- Shared memory tiling can yield **>10× improvement** over the naive kernel.
- True peak performance requires advanced optimizations (e.g., loop unrolling, tensor cores, or cuBLAS comparison).

------

## 🧰 Environment

| Component    | Version          |
| ------------ | ---------------- |
| CUDA Toolkit | 12.x             |
| Driver       | 550+             |
| Compiler     | `nvcc`           |
| OS           | Ubuntu 20.04 LTS |
| GPU          | GeForce RTX 4060 |

------

## 👨‍💻 Author

**Yangyang Zhang**
 M.Eng. in Computing & Software, McMaster University
 📧 [zhang787@mcmaster.ca](mailto:zhang787@mcmaster.ca) | 🌐 [LinkedIn](https://linkedin.com/in/) | [GitHub](https://github.com/)

