# GPU-X

## CUDA GPU Performance Optimization & Benchmarking

GPU-X is a CUDA-based GPU performance engineering project focused on matrix multiplication optimization, GPU memory behavior, kernel tuning, benchmarking, and NVIDIA Nsight Compute profiling.

The project progressively transforms a baseline CPU implementation into an optimized CUDA implementation and evaluates the impact of different GPU optimization techniques.

---

## Project Highlights

* CUDA-based matrix multiplication
* CPU vs GPU benchmarking
* Naive CUDA implementation
* Shared memory optimization
* Tiled matrix multiplication
* Tile-size benchmarking
* Block configuration experiments
* Register optimization
* Fast math optimization
* NVIDIA Nsight Compute profiling
* GPU occupancy analysis
* GPU throughput analysis
* Performance visualization
* Numerical result verification

---

## Hardware

| Component          | Specification           |
| ------------------ | ----------------------- |
| GPU                | NVIDIA GeForce RTX 2050 |
| Compute Capability | 8.6                     |
| CUDA Architecture  | Ampere                  |
| SMs                | 16                      |
| Matrix Size        | 1024 x 1024             |
| Best Tile Size     | 16 x 16                 |

---

## Project Architecture

```text
                         GPU-X
                           |
            +--------------+--------------+
            |                             |
       CUDA Kernels                 Benchmarking
            |                             |
      +-----+--------+              +-----+--------+
      |     |        |              |     |        |
    Naive Tiled  Optimized         CPU   GPU   Profiling
      |     |        |              |     |        |
      +-----+--------+              +-----+--------+
            |                             |
            +--------------+--------------+
                           |
                    Performance Data
                           |
                     Visualization
                           |
                     Final Analysis
```

---

# Optimization Journey

GPU-X was developed through multiple optimization phases.

### Phase 1-3 - CUDA Fundamentals

Established the CUDA environment and implemented initial GPU kernels.

### Phase 4 - Shared Memory

Compared global-memory access with shared-memory access and studied the GPU memory hierarchy.

### Phase 5 - Matrix Multiplication

Implemented the baseline CUDA matrix multiplication kernel.

### Phase 6 - Tile Benchmarking

Compared multiple tile configurations including:

```text
8 x 8
16 x 16
32 x 32
```

The 16 x 16 configuration produced the best overall result.

### Phase 7 - Kernel Optimization

Applied kernel-level optimization techniques including:

* Memory access improvements
* Register tuning
* Occupancy analysis
* Kernel configuration tuning

### Phase 8 - Mathematical Optimization

Experimented with:

* Fast math
* Compiler optimizations
* Register constraints

### Phase 9 - Nsight Compute Profiling

The optimized kernel was analyzed using NVIDIA Nsight Compute.

Important metrics included:

* GPU throughput
* Memory throughput
* DRAM throughput
* L1/TEX throughput
* Occupancy
* Registers per thread
* SM utilization

### Phase 10 - Final Benchmark

A complete CPU vs GPU benchmark was executed with numerical verification.

### Phase 11 - Visualization

Performance and profiling results were converted into graphs for analysis and presentation.

---

# Final Benchmark

Matrix size:

```text
1024 x 1024
```

### Execution Time

| Implementation |         Time |
| -------------- | -----------: |
| CPU            | 5119.9697 ms |
| Naive GPU      |    5.8519 ms |
| Optimized GPU  |    4.4236 ms |

---

# Speedup

### CPU to Naive GPU

```text
874.9190x
```

### CPU to Optimized GPU

```text
1157.4257x
```

### Naive GPU to Optimized GPU

```text
1.3229x
```

### Optimization Improvement

```text
24.4082%
```

The optimized CUDA implementation achieved more than 1100x speedup over the CPU baseline for the tested workload.

---

# Nsight Compute Results

The optimized kernel was profiled on the NVIDIA RTX 2050.

| Metric                | Result |
| --------------------- | -----: |
| Compute Throughput    | 98.49% |
| Memory Throughput     | 98.49% |
| L1/TEX Throughput     | 98.55% |
| DRAM Throughput       | 57.31% |
| Theoretical Occupancy |   100% |
| Achieved Occupancy    | 98.97% |
| Registers per Thread  |     40 |

These measurements show that the optimized kernel makes strong use of the available GPU resources.

---

# Verification

Every major implementation was numerically verified.

```text
CPU Verification       : PASSED
Naive GPU Verification : PASSED
Tiled Verification     : PASSED
Final Verification     : PASSED
```

The verification step ensures that GPU optimization did not change the mathematical result.

---

# Results and Graphs

Performance visualizations are available under:

```text
results/graphs/
```

Current graphs include:

```text
cpu_vs_gpu.png
tile_comparison.png
optimization_progress.png
speedup_comparison.png
nsight_occupancy.png
nsight_throughput.png
```

These graphs provide visual comparisons of:

* CPU vs GPU performance
* CUDA tile-size performance
* Optimization progression
* GPU speedup
* GPU occupancy
* GPU throughput

---

# Project Structure

```text
GPU-X/
|
+-- benchmark/
|   +-- final_benchmark.cu
|   +-- memory_transfer.cu
|   +-- vector_add.cu
|
+-- optimization/
|   +-- matrix/
|   +-- block_config/
|   +-- ...
|
+-- results/
|   +-- benchmark_results.txt
|   +-- generate_graphs.py
|   +-- nsight_graphs.py
|   |
|   +-- graphs/
|       +-- cpu_vs_gpu.png
|       +-- tile_comparison.png
|       +-- optimization_progress.png
|       +-- speedup_comparison.png
|       +-- nsight_occupancy.png
|       +-- nsight_throughput.png
|
+-- build/
|
+-- docs/
|   +-- GPU-X_Technical_Report.md
|
+-- README.md
```

---

# Technologies

* C++
* CUDA C++
* NVIDIA CUDA Toolkit
* NVIDIA Nsight Compute
* PowerShell
* Python
* Matplotlib

---

# Project Objectives

The main objectives of GPU-X are:

1. Understand CUDA GPU programming.
2. Compare CPU and GPU computation.
3. Understand GPU memory hierarchy.
4. Optimize memory access patterns.
5. Implement shared-memory tiling.
6. Study thread-block configurations.
7. Study register usage and occupancy.
8. Profile GPU kernels using Nsight Compute.
9. Measure real-world performance improvements.
10. Build a reproducible GPU performance benchmark.

---

# Future Scope

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
* Automated GPU parameter tuning
* Cross-GPU benchmarking

---

# Conclusion

GPU-X demonstrates how CUDA optimization techniques can significantly improve computational performance.

Starting from a CPU matrix multiplication implementation, the project progressively introduced GPU acceleration, tiled matrix multiplication, memory optimization, register tuning, mathematical optimization, and GPU profiling.

For the tested 1024 x 1024 matrix workload, the final benchmark achieved:

```text
CPU Time            : 5119.9697 ms
Optimized GPU Time  : 4.4236 ms
CPU to Optimized GPU: 1157.4257x
```

The project also achieved:

```text
Achieved Occupancy : 98.97%
Compute Throughput : 98.49%
Memory Throughput  : 98.49%
```

according to NVIDIA Nsight Compute profiling.

GPU-X demonstrates the practical impact of GPU architecture-aware optimization and provides a foundation for further CUDA performance engineering.

---

## Author

**Rohan Raj Chaudhary**

GPU Performance Engineering Project

CUDA | C++ | NVIDIA RTX 2050
