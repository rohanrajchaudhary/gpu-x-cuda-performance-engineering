# ⚡ GPU-X

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://gpu-x-cuda-performance-engineering.vercel.app/)
[![GitHub Repo](https://img.shields.io/badge/📦_Repository-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rohanrajchaudhary/gpu-x-cuda-performance-engineering)

* **🌐 Live Web App:** [gpu-x-cuda-performance-engineering.vercel.app](https://gpu-x-cuda-performance-engineering.vercel.app/)
* **📂 GitHub Repository:** [github.com/rohanrajchaudhary/gpu-x-cuda-performance-engineering](https://github.com/rohanrajchaudhary/gpu-x-cuda-performance-engineering)

---

## CUDA GPU Performance Optimization & Benchmarking

**GPU-X** is a CUDA-based GPU performance engineering project focused on **matrix multiplication optimization, GPU memory behavior, kernel tuning, benchmarking, and NVIDIA Nsight Compute profiling**.

The project starts from a CPU baseline and progressively explores CUDA acceleration, tiled matrix multiplication, block configuration, register tuning, fast-math optimization, profiling, and performance visualization.

---

## 🚀 Key Results

Test workload: **1024 × 1024 matrix multiplication**

| Implementation     |   Execution Time |
| ------------------ | ---------------: |
| CPU                | **5119.9697 ms** |
| Naive CUDA GPU     |    **5.8519 ms** |
| Optimized CUDA GPU |    **4.4236 ms** |

### Performance

| Comparison                |         Result |
| ------------------------- | -------------: |
| CPU → Naive GPU           |  **874.9190×** |
| CPU → Optimized GPU       | **1157.4257×** |
| Naive GPU → Optimized GPU |    **1.3229×** |
| Optimization improvement  |   **24.4082%** |

All implementations were numerically verified successfully.

---

## 🖥️ Hardware & CUDA Environment

| Component          | Configuration           |
| ------------------ | ----------------------- |
| GPU                | NVIDIA GeForce RTX 2050 |
| GPU Memory         | 4 GB                    |
| Architecture       | NVIDIA Ampere           |
| Compute Capability | 8.6 (`sm_86`)           |
| SMs                | 16                      |
| Matrix Size        | 1024 × 1024             |
| Best Tile Size     | 16 × 16                 |
| CUDA Toolkit       | 13.3                    |
| Compiler           | `nvcc` + MSVC x64       |

The CUDA environment and first GPU kernel were successfully validated on the RTX 2050 before beginning the optimization experiments.

---

# 🧠 Optimization Journey

GPU-X was developed incrementally to study the effect of different GPU optimization techniques.

### Phase 1–3 — CUDA Fundamentals

* CUDA environment setup
* GPU detection
* CUDA compiler validation
* First GPU kernel
* Basic parallel workload experiments

The initial environment validation confirmed the RTX 2050, CUDA Toolkit, `nvcc`, MSVC x64 compiler, kernel compilation, and GPU execution.

### Phase 4 — GPU Memory

Explored GPU memory behavior and shared-memory based computation.

### Phase 5 — Matrix Multiplication

Implemented CUDA matrix multiplication and established the tiled implementation.

### Phase 6 — Tile Benchmarking

Different block/tile configurations were benchmarked to identify a strong configuration for the workload.

The final selected configuration was:

```text
16 × 16
```

### Phase 7 — Kernel Optimization

Investigated:

* Memory access optimization
* Register usage
* Occupancy
* Kernel configuration
* Thread-block behavior

### Phase 8 — Mathematical & Register Optimization

Investigated:

* Fast math
* Compiler optimization
* Register constraints
* Kernel-level tuning

The best measured kernel time during the optimization journey reached approximately:

```text
4.4145 ms
```

### Phase 9 — Nsight Compute Profiling

The optimized kernel was analyzed using NVIDIA Nsight Compute.

Important metrics included:

* Compute throughput
* Memory throughput
* DRAM throughput
* L1/TEX throughput
* Occupancy
* Registers per thread
* SM utilization

### Phase 10 — Final Benchmark

A complete CPU vs GPU benchmark was executed with numerical verification.

### Phase 11 — Visualization

Benchmark and profiling results were converted into graphs for easier performance analysis.

---

# 🔬 GPU Architecture & Kernel Strategy

The main matrix multiplication implementation uses a **2D CUDA thread block** with shared-memory tiles.

Conceptually:

```text
Global Memory
     │
     ▼
┌───────────────┐
│   Tile A      │
│   Shared Mem  │
└───────────────┘
        ×
┌───────────────┐
│   Tile B      │
│   Shared Mem  │
└───────────────┘
        │
        ▼
   Accumulation
        │
        ▼
   Global Memory
       C
```

Each thread contributes to the computation of an output matrix element while the kernel reuses data loaded into shared memory.

---

# 📊 Nsight Compute Analysis

Profiling was performed on the optimized tiled matrix multiplication kernel.

### GPU Throughput

| Metric             |     Result |
| ------------------ | ---------: |
| Compute Throughput | **98.49%** |
| Memory Throughput  | **98.49%** |
| L1/TEX Throughput  | **98.55%** |
| DRAM Throughput    | **57.31%** |
| L2 Throughput      | **27.28%** |

### Occupancy

| Metric                     |          Result |
| -------------------------- | --------------: |
| Theoretical Occupancy      |        **100%** |
| Achieved Occupancy         |      **98.97%** |
| Achieved Active Warps / SM |       **47.50** |
| Registers / Thread         |          **40** |
| Block Size                 | **256 threads** |

These measurements indicate that the workload makes strong use of the GPU's compute and memory resources.

---

# 📈 Results & Visualizations

Performance graphs are available in:

```text
results/graphs/
```

Current visualizations:

```text
cpu_vs_gpu.png
tile_comparison.png
optimization_progress.png
speedup_comparison.png
nsight_occupancy.png
nsight_throughput.png
```

### Visualizations include

* CPU vs GPU execution time
* Tile-size comparison
* Optimization progression
* CPU/GPU speedup
* Nsight Compute occupancy
* Nsight Compute throughput

---

# 🧪 Verification

Numerical correctness was checked after each major implementation.

```text
CPU Verification       : PASSED
Naive GPU Verification : PASSED
Tiled Verification     : PASSED
Final Verification     : PASSED
```

Performance optimization was therefore evaluated together with correctness rather than relying only on execution time.

---

# 📁 Project Structure

```text
GPU-X/
│
├── benchmark/
│   ├── final_benchmark.cu
│   ├── memory_transfer.cu
│   └── vector_add.cu
│
├── memory/
│   └── shared_memory.cu
│
├── optimization/
│   ├── matrix_tiling.cu
│   ├── matrix_tiling_opt.cu
│   ├── matrix_tiling_reg.cu
│   ├── matrix_tiling_vec.cu
│   ├── tile_benchmark.cu
│   ├── block_config_benchmark.cu
│   ├── phase8b_fastmath.cu
│   ├── phase8c_register.cu
│   ├── phase8d_register40.cu
│   └── phase9_profiling.cu
│
├── results/
│   ├── benchmark_results.txt
│   ├── generate_graphs.py
│   ├── nsight_graphs.py
│   └── graphs/
│
├── docs/
│   └── GPU-X_Technical_Report.md
│
├── hello_gpu.cu
├── .gitignore
└── README.md
```

---

# 🛠️ Technologies

* **C++**
* **CUDA C++**
* **NVIDIA CUDA Toolkit**
* **NVIDIA Nsight Compute**
* **Python**
* **Matplotlib**
* **PowerShell**
* **Git / GitHub**

---

# 🎯 Project Objectives

1. Understand CUDA GPU programming.
2. Compare CPU and GPU computation.
3. Study GPU memory hierarchy.
4. Optimize GPU memory access.
5. Implement shared-memory tiling.
6. Benchmark different tile configurations.
7. Study register usage and occupancy.
8. Profile CUDA kernels with Nsight Compute.
9. Measure practical performance improvements.
10. Build a reproducible GPU performance benchmarking workflow.

---

# ▶️ Build & Run

GPU-X targets NVIDIA GPUs with CUDA support.

Example CUDA compilation:

```powershell
nvcc -O3 -arch=sm_86 <source>.cu -o <output>.exe
```

For the Windows development environment used during the project, `nvcc` was configured with the x64 MSVC host compiler.

Example:

```powershell
nvcc -O3 -arch=sm_86 -ccbin "<MSVC-x64-path>" <source>.cu -o <output>.exe
```

Run the generated executable:

```powershell
.\<output>.exe
```

---

# 🔍 Profiling

NVIDIA Nsight Compute can be used to inspect the CUDA kernels and collect metrics such as:

* Compute throughput
* Memory throughput
* Occupancy
* Register usage
* L1/L2 behavior
* DRAM utilization
* SM activity

This allows optimization decisions to be based on GPU profiling data rather than execution time alone.

---

# 🔮 Future Scope

GPU-X can be extended with:

* CUDA streams
* Asynchronous memory transfers
* Pinned host memory
* CUDA Graphs
* Multi-GPU execution
* cuBLAS comparison
* Tensor Core optimization
* Half-precision computation
* Mixed-precision matrix multiplication
* Larger matrix workloads
* Automated kernel parameter tuning
* Cross-GPU benchmarking

---

# 🏁 Conclusion

GPU-X demonstrates the practical impact of GPU architecture-aware optimization using CUDA.

Starting from a CPU matrix multiplication baseline, the project progressively introduced GPU acceleration, tiled matrix multiplication, memory optimization, block configuration experiments, register tuning, fast-math optimization, Nsight Compute profiling, and performance visualization.

For the tested **1024 × 1024 matrix workload**, the final benchmark measured:

```text
CPU Time           : 5119.9697 ms
Naive GPU Time     : 5.8519 ms
Optimized GPU Time : 4.4236 ms
```

Resulting in:

```text
CPU → Optimized GPU : 1157.4257×
Optimization Gain   : 24.4082%
```

The profiling analysis also recorded approximately **98.97% achieved occupancy** and **98.49% compute throughput** for the profiled workload.

GPU-X provides a practical foundation for further CUDA performance engineering and GPU architecture experimentation.

---

## 👨‍💻 Author

**Rohan Raj Chaudhary**

**GPU Performance Engineering Project**

`CUDA • C++ • NVIDIA RTX 2050 • Nsight Compute`
