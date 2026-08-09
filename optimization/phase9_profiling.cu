#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define N 1024
#define TILE 16

__global__ void tiledMatrixMul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

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
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main()
{
    const size_t bytes =
        static_cast<size_t>(N) * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    // Warm-up
    tiledMatrixMul<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    tiledMatrixMul<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaMemcpy(
        h_C,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    bool correct = true;

    for (int i = 0; i < 10; ++i)
    {
        if (h_C[i] != 2048.0f)
        {
            correct = false;
            break;
        }
    }

    std::cout << std::fixed << std::setprecision(4);

    std::cout
        << "\n========================================\n"
        << "       GPU-X PHASE 9 PROFILING\n"
        << "========================================\n\n";

    std::cout
        << "Matrix Size : "
        << N << " x " << N << "\n";

    std::cout
        << "Tile Size   : "
        << TILE << " x " << TILE << "\n";

    std::cout
        << "Kernel Time : "
        << milliseconds << " ms\n";

    std::cout
        << "Baseline Best : 4.4145 ms\n";

    std::cout
        << "\nVerification: "
        << (correct ? "PASSED" : "FAILED")
        << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}