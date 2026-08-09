#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void vectorAddGPU(
    const float* A,
    const float* B,
    float* C,
    int N
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

void vectorAddCPU(
    const float* A,
    const float* B,
    float* C,
    int N
)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1 << 24;

    size_t bytes = N * sizeof(float);

    std::cout << "=====================================\n";
    std::cout << "       GPU-X VECTOR BENCHMARK\n";
    std::cout << "=====================================\n";

    std::cout << "Elements: " << N << "\n";
    std::cout << "Memory per vector: "
              << bytes / (1024.0 * 1024.0)
              << " MB\n\n";

    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);

    // =========================
    // CPU
    // =========================

    auto cpuStart = std::chrono::high_resolution_clock::now();

    vectorAddCPU(
        h_A.data(),
        h_B.data(),
        h_C.data(),
        N
    );

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart
        ).count();

    std::cout << "CPU Time: "
              << cpuTime
              << " ms\n";


    // =========================
    // GPU MEMORY
    // =========================

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(
        d_A,
        h_A.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B.data(),
        bytes,
        cudaMemcpyHostToDevice
    );


    // =========================
    // GPU KERNEL
    // =========================

    int threads = 256;

    int blocks =
        (N + threads - 1) / threads;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vectorAddGPU<<<blocks, threads>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float gpuTime = 0;

    cudaEventElapsedTime(
        &gpuTime,
        start,
        stop
    );

    std::cout << "GPU Kernel Time: "
              << gpuTime
              << " ms\n";


    // =========================
    // COPY RESULT BACK
    // =========================

    cudaMemcpy(
        h_C.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // =========================
    // VERIFY
    // =========================

    bool correct = true;

    for (int i = 0; i < 10; i++)
    {
        if (h_C[i] != 3.0f)
        {
            correct = false;
            break;
        }
    }

    std::cout << "\nResult Verification: "
              << (correct ? "PASSED" : "FAILED")
              << "\n";


    // =========================
    // SPEEDUP
    // =========================

    double speedup =
        cpuTime / gpuTime;

    std::cout << "\nGPU Speedup: "
              << speedup
              << "x\n";


    // =========================
    // CLEANUP
    // =========================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}