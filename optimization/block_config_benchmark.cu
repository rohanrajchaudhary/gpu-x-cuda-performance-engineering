#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define N 1024

__global__ void matrixMul8(
    const float* A,
    const float* B,
    float* C)
{
    __shared__ float As[8][8];
    __shared__ float Bs[8][8];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * 8 + ty;
    int col = blockIdx.x * 8 + tx;

    float sum = 0.0f;

    for (int t = 0; t < N / 8; t++)
    {
        As[ty][tx] = A[row * N + t * 8 + tx];
        Bs[ty][tx] = B[(t * 8 + ty) * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}


__global__ void matrixMul16(
    const float* A,
    const float* B,
    float* C)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;

    float sum = 0.0f;

    for (int t = 0; t < N / 16; t++)
    {
        As[ty][tx] = A[row * N + t * 16 + tx];
        Bs[ty][tx] = B[(t * 16 + ty) * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; k++)
        {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}


__global__ void matrixMul32x8(
    const float* A,
    const float* B,
    float* C)
{
    __shared__ float As[8][32];
    __shared__ float Bs[8][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * 8 + ty;
    int col = blockIdx.x * 32 + tx;

    float sum = 0.0f;

    for (int t = 0; t < N / 8; t++)
    {
        As[ty][tx] = A[row * N + t * 8 + tx];
        Bs[ty][tx] = B[(t * 8 + ty) * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}


float runTile8(float* A, float* B, float* C)
{
    dim3 block(8, 8);
    dim3 grid(N / 8, N / 8);

    matrixMul8<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMul8<<<grid, block>>>(A, B, C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}


float runTile16(float* A, float* B, float* C)
{
    dim3 block(16, 16);
    dim3 grid(N / 16, N / 16);

    matrixMul16<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMul16<<<grid, block>>>(A, B, C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}


float runTile32x8(float* A, float* B, float* C)
{
    dim3 block(32, 8);
    dim3 grid(N / 32, N / 8);

    matrixMul32x8<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMul32x8<<<grid, block>>>(A, B, C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}


int main()
{
    size_t bytes =
        (size_t)N * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice
    );

    std::cout << std::fixed
              << std::setprecision(4);

    std::cout
        << "\n========================================\n"
        << "     GPU-X PHASE 8A BLOCK BENCHMARK\n"
        << "========================================\n\n";

    float t8 = runTile8(
        d_A,
        d_B,
        d_C
    );

    std::cout
        << "TILE 8x8   : "
        << t8
        << " ms\n";

    float t16 = runTile16(
        d_A,
        d_B,
        d_C
    );

    std::cout
        << "TILE 16x16 : "
        << t16
        << " ms\n";

    float t32x8 = runTile32x8(
        d_A,
        d_B,
        d_C
    );

    std::cout
        << "TILE 32x8  : "
        << t32x8
        << " ms\n";

    float best = t8;
    const char* bestName = "8x8";

    if (t16 < best)
    {
        best = t16;
        bestName = "16x16";
    }

    if (t32x8 < best)
    {
        best = t32x8;
        bestName = "32x8";
    }

    cudaMemcpy(
        h_C,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    bool correct = true;

    for (int i = 0; i < 10; i++)
    {
        if (h_C[i] != 2048.0f)
        {
            correct = false;
            break;
        }
    }

    std::cout
        << "\n----------------------------------------\n"
        << "BEST CONFIG : "
        << bestName
        << "\n";

    std::cout
        << "BEST TIME   : "
        << best
        << " ms\n";

    std::cout
        << "----------------------------------------\n";

    std::cout
        << "\nVerification: "
        << (correct ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "\nCurrent Project Best : 4.4257 ms\n";

    if (best < 4.4257f)
    {
        std::cout
            << "Optimization: NEW BEST!\n";
    }
    else
    {
        std::cout
            << "Optimization: Current best remains faster.\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}