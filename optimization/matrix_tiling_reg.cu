#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define N 1024
#define TILE 16

// Modified Kernel with Launch Bounds and Register Optimizations
__global__ __launch_bounds__(256, 2)
void tiledMatrixMulReg(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * 16 + ty;
    const int col = blockIdx.x * 16 + tx;

    float sum = 0.0f;

    #pragma unroll 4
    for (int t = 0; t < 64; ++t)
    {
        tileA[ty][tx] = A[row * 1024 + t * 16 + tx];
        tileB[ty][tx] = B[(t * 16 + ty) * 1024 + col];

        __syncthreads();

        #pragma unroll 4
        for (int k = 0; k < 16; ++k)
        {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    C[row * 1024 + col] = sum;
}

int main()
{
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // Initialization
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

    // Warm-up launch using new kernel
    tiledMatrixMulReg<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timed Kernel Execution
    cudaEventRecord(start);

    tiledMatrixMulReg<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verification against 2048.0f
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
        << "    GPU-X PHASE 7C REGISTER OPTIMIZATION\n"
        << "========================================\n\n";

    std::cout << "Matrix Size : " << N << " x " << N << "\n";
    std::cout << "Tile Size   : " << TILE << "\n";
    std::cout << "Kernel Time : " << milliseconds << " ms\n";

    std::cout << "\nVerification: " << (correct ? "PASSED" : "FAILED") << "\n";

    std::cout << "\nCurrent Best : 4.4421 ms\n";

    if (milliseconds < 4.4421f)
    {
        std::cout << "Optimization: NEW BEST!\n";
    }
    else
    {
        std::cout << "Optimization: Current best remains faster.\n";
    }

    // Cleanup
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