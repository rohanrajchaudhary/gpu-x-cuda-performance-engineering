#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define N 1024

// =====================================================
// TILED MATRIX MULTIPLICATION
// =====================================================

template<int TILE>
__global__ void tiledMatrixMul(
    const float* A,
    const float* B,
    float* C
)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++)
    {
        tileA[threadIdx.y][threadIdx.x] =
            A[row * N + t * TILE + threadIdx.x];

        tileB[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            sum +=
                tileA[threadIdx.y][k] *
                tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}


// =====================================================
// BENCHMARK FUNCTION
// =====================================================

template<int TILE>
float benchmarkTile(
    float* d_A,
    float* d_B,
    float* d_C
)
{
    dim3 threads(TILE, TILE);

    dim3 blocks(
        N / TILE,
        N / TILE
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    tiledMatrixMul<TILE>
        <<<blocks, threads>>>(
            d_A,
            d_B,
            d_C
        );

    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start);

    tiledMatrixMul<TILE>
        <<<blocks, threads>>>(
            d_A,
            d_B,
            d_C
        );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}


// =====================================================
// MAIN
// =====================================================

int main()
{
    const size_t bytes =
        static_cast<size_t>(N) *
        N *
        sizeof(float);

    std::cout
        << "========================================\n";

    std::cout
        << "       GPU-X TILE SIZE BENCHMARK\n";

    std::cout
        << "========================================\n\n";

    std::cout
        << "Matrix: "
        << N
        << " x "
        << N
        << "\n\n";


    // =================================================
    // HOST DATA
    // =================================================

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }


    // =================================================
    // GPU MEMORY
    // =================================================

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
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice
    );


    // =================================================
    // TILE 8
    // =================================================

    float time8 =
        benchmarkTile<8>(
            d_A,
            d_B,
            d_C
        );


    // =================================================
    // TILE 16
    // =================================================

    float time16 =
        benchmarkTile<16>(
            d_A,
            d_B,
            d_C
        );


    // =================================================
    // TILE 32
    // =================================================

    float time32 =
        benchmarkTile<32>(
            d_A,
            d_B,
            d_C
        );


    // =================================================
    // COPY RESULT
    // =================================================

    cudaMemcpy(
        h_C,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // =================================================
    // VERIFY
    // =================================================

    bool correct = true;

    for (int i = 0; i < 10; i++)
    {
        if (h_C[i] != 2048.0f)
        {
            correct = false;
            break;
        }
    }


    // =================================================
    // RESULTS
    // =================================================

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "\n--------------- RESULTS ---------------\n\n";

    std::cout
        << "TILE 8  : "
        << time8
        << " ms\n";

    std::cout
        << "TILE 16 : "
        << time16
        << " ms\n";

    std::cout
        << "TILE 32 : "
        << time32
        << " ms\n";


    // =================================================
    // BEST TILE
    // =================================================

    float bestTime = time8;
    int bestTile = 8;

    if (time16 < bestTime)
    {
        bestTime = time16;
        bestTile = 16;
    }

    if (time32 < bestTime)
    {
        bestTime = time32;
        bestTile = 32;
    }

    std::cout
        << "\n----------------------------------------\n";

    std::cout
        << "BEST TILE SIZE : "
        << bestTile
        << "\n";

    std::cout
        << "BEST TIME      : "
        << bestTime
        << " ms\n";

    std::cout
        << "\nVerification: "
        << (correct ? "PASSED" : "FAILED")
        << "\n";


    // =================================================
    // CLEANUP
    // =================================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}