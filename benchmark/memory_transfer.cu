#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

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

int main()
{
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    std::cout << "========================================\n";
    std::cout << "      GPU-X MEMORY TRANSFER BENCHMARK\n";
    std::cout << "========================================\n\n";

    std::cout << "Elements: " << N << "\n";
    std::cout << "Data size per vector: "
              << bytes / (1024.0 * 1024.0)
              << " MB\n\n";

    // Host memory
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);

    // Device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // CUDA events
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float h2dTime = 0.0f;
    float kernelTime = 0.0f;
    float d2hTime = 0.0f;

    // ==========================================
    // 1. HOST → DEVICE
    // ==========================================

    cudaEventRecord(start);

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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(
        &h2dTime,
        start,
        stop
    );


    // ==========================================
    // 2. GPU KERNEL
    // ==========================================

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEventRecord(start);

    vectorAddGPU<<<blocks, threads>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(
        &kernelTime,
        start,
        stop
    );


    // ==========================================
    // 3. DEVICE → HOST
    // ==========================================

    cudaEventRecord(start);

    cudaMemcpy(
        h_C.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(
        &d2hTime,
        start,
        stop
    );


    // ==========================================
    // TOTAL GPU TIME
    // ==========================================

    float totalGPUTime =
        h2dTime +
        kernelTime +
        d2hTime;


    // ==========================================
    // VERIFY RESULT
    // ==========================================

    bool correct = true;

    for (int i = 0; i < 10; i++)
    {
        if (h_C[i] != 3.0f)
        {
            correct = false;
            break;
        }
    }


    // ==========================================
    // RESULTS
    // ==========================================

    std::cout << std::fixed
              << std::setprecision(4);

    std::cout << "\n--------------- RESULTS ---------------\n\n";

    std::cout << "Host -> Device : "
              << h2dTime
              << " ms\n";

    std::cout << "GPU Kernel     : "
              << kernelTime
              << " ms\n";

    std::cout << "Device -> Host : "
              << d2hTime
              << " ms\n";

    std::cout << "----------------------------------------\n";

    std::cout << "Total GPU Time : "
              << totalGPUTime
              << " ms\n";

    std::cout << "\nResult Verification: "
              << (correct ? "PASSED" : "FAILED")
              << "\n";


    // ==========================================
    // PERCENTAGE BREAKDOWN
    // ==========================================

    double transferTime =
        h2dTime + d2hTime;

    double transferPercentage =
        (transferTime / totalGPUTime) * 100.0;

    double kernelPercentage =
        (kernelTime / totalGPUTime) * 100.0;

    std::cout << "\n----------- TIME BREAKDOWN -------------\n\n";

    std::cout << "Memory Transfer: "
              << transferPercentage
              << "%\n";

    std::cout << "GPU Compute:     "
              << kernelPercentage
              << "%\n";


    // Cleanup

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}