#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

#define N 1024
#define TILE 16

// ============================================================
// CPU MATRIX MULTIPLICATION
// ============================================================

void cpuMatrixMul(
    const float* A,
    const float* B,
    float* C)
{
    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; ++k)
            {
                sum += A[row * N + k] *
                       B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

// ============================================================
// NAIVE GPU MATRIX MULTIPLICATION
// ============================================================

__global__ void naiveMatrixMul(
    const float* A,
    const float* B,
    float* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k)
        {
            sum += A[row * N + k] *
                   B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// ============================================================
// OPTIMIZED TILED GPU MATRIX MULTIPLICATION
// ============================================================

__global__ void tiledMatrixMul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    #pragma unroll
    for (int t = 0; t < N / TILE; ++t)
    {
        tileA[ty][tx] =
            A[row * N + t * TILE + tx];

        tileB[ty][tx] =
            B[(t * TILE + ty) * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
        {
            sum += tileA[ty][k] *
                   tileB[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

// ============================================================
// GPU TIMING HELPER
// ============================================================

float benchmarkNaiveGPU(
    float* d_A,
    float* d_B,
    float* d_C)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    // Warm-up
    naiveMatrixMul<<<blocks, threads>>>(
        d_A, d_B, d_C);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    naiveMatrixMul<<<blocks, threads>>>(
        d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// ============================================================
// OPTIMIZED GPU BENCHMARK
// ============================================================

float benchmarkTiledGPU(
    float* d_A,
    float* d_B,
    float* d_C)
{
    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    // Warm-up
    tiledMatrixMul<<<blocks, threads>>>(
        d_A, d_B, d_C);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    tiledMatrixMul<<<blocks, threads>>>(
        d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// ============================================================
// VERIFICATION
// ============================================================

bool verifyResult(
    const float* C)
{
    const float expected = 2048.0f;

    for (int i = 0; i < N * N; ++i)
    {
        if (std::fabs(C[i] - expected) > 0.01f)
        {
            return false;
        }
    }

    return true;
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    std::cout << std::fixed
              << std::setprecision(4);

    const size_t elements =
        static_cast<size_t>(N) * N;

    const size_t bytes =
        elements * sizeof(float);

    // --------------------------------------------------------
    // HOST MEMORY
    // --------------------------------------------------------

    float* h_A = new float[elements];
    float* h_B = new float[elements];
    float* h_C = new float[elements];

    for (size_t i = 0; i < elements; ++i)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // --------------------------------------------------------
    // CPU BENCHMARK
    // --------------------------------------------------------

    auto cpuStart =
        std::chrono::high_resolution_clock::now();

    cpuMatrixMul(
        h_A,
        h_B,
        h_C);

    auto cpuEnd =
        std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart).count();

    bool cpuCorrect =
        verifyResult(h_C);

    // --------------------------------------------------------
    // DEVICE MEMORY
    // --------------------------------------------------------

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice);

    // --------------------------------------------------------
    // NAIVE GPU
    // --------------------------------------------------------

    float naiveTime =
        benchmarkNaiveGPU(
            d_A,
            d_B,
            d_C);

    cudaMemcpy(
        h_C,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost);

    bool naiveCorrect =
        verifyResult(h_C);

    // --------------------------------------------------------
    // OPTIMIZED GPU
    // --------------------------------------------------------

    float tiledTime =
        benchmarkTiledGPU(
            d_A,
            d_B,
            d_C);

    cudaMemcpy(
        h_C,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost);

    bool tiledCorrect =
        verifyResult(h_C);

    // --------------------------------------------------------
    // SPEEDUPS
    // --------------------------------------------------------

    double cpuToNaive =
        cpuTime / naiveTime;

    double cpuToTiled =
        cpuTime / tiledTime;

    double naiveToTiled =
        naiveTime / tiledTime;

    double improvement =
        ((naiveTime - tiledTime) /
         naiveTime) * 100.0;

    // --------------------------------------------------------
    // FINAL REPORT
    // --------------------------------------------------------

    std::cout
        << "\n========================================\n"
        << "       GPU-X PHASE 10\n"
        << "    FINAL PERFORMANCE BENCHMARK\n"
        << "========================================\n\n";

    std::cout
        << "Matrix Size : "
        << N << " x " << N << "\n";

    std::cout
        << "Tile Size   : "
        << TILE << " x " << TILE << "\n";

    std::cout
        << "\n--------------- RESULTS ---------------\n\n";

    std::cout
        << "CPU Time        : "
        << cpuTime
        << " ms\n";

    std::cout
        << "Naive GPU Time  : "
        << naiveTime
        << " ms\n";

    std::cout
        << "Optimized GPU   : "
        << tiledTime
        << " ms\n";

    std::cout
        << "\n--------------- SPEEDUP ---------------\n\n";

    std::cout
        << "CPU -> Naive GPU : "
        << cpuToNaive
        << "x\n";

    std::cout
        << "CPU -> Optimized : "
        << cpuToTiled
        << "x\n";

    std::cout
        << "Naive -> Optimized : "
        << naiveToTiled
        << "x\n";

    std::cout
        << "\nOptimization Improvement : "
        << improvement
        << "%\n";

    std::cout
        << "\n------------- VERIFICATION ------------\n\n";

    std::cout
        << "CPU Verification   : "
        << (cpuCorrect ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "Naive Verification : "
        << (naiveCorrect ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "Tiled Verification : "
        << (tiledCorrect ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "\n========================================\n";

    // --------------------------------------------------------
    // CLEANUP
    // --------------------------------------------------------

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}