#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

#define N 1024
#define TILE 16

// ======================================================
// CPU MATRIX MULTIPLICATION
// ======================================================

void cpuMatrixMul(
    const float* A,
    const float* B,
    float* C
)
{
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
            {
                sum += A[row * N + k] *
                       B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}


// ======================================================
// NAIVE CUDA MATRIX MULTIPLICATION
// ======================================================

__global__ void naiveMatrixMul(
    const float* A,
    const float* B,
    float* C
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] *
                   B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// ======================================================
// TILED CUDA MATRIX MULTIPLICATION
// ======================================================

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

    for (int tile = 0; tile < N / TILE; tile++)
    {
        // Load A tile
        tileA[threadIdx.y][threadIdx.x] =
            A[row * N + tile * TILE + threadIdx.x];

        // Load B tile
        tileB[threadIdx.y][threadIdx.x] =
            B[(tile * TILE + threadIdx.y) * N + col];

        // Wait for all threads
        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE; k++)
        {
            sum +=
                tileA[threadIdx.y][k] *
                tileB[k][threadIdx.x];
        }

        // Wait before loading next tile
        __syncthreads();
    }

    C[row * N + col] = sum;
}


// ======================================================
// VERIFY RESULTS
// ======================================================

bool verify(
    const float* reference,
    const float* result
)
{
    for (int i = 0; i < N * N; i++)
    {
        float difference =
            fabs(reference[i] - result[i]);

        if (difference > 0.01f)
        {
            return false;
        }
    }

    return true;
}


// ======================================================
// MAIN
// ======================================================

int main()
{
    const size_t bytes =
        static_cast<size_t>(N) *
        N *
        sizeof(float);

    std::cout
        << "========================================\n";

    std::cout
        << "     GPU-X MATRIX MULTIPLICATION\n";

    std::cout
        << "========================================\n\n";

    std::cout
        << "Matrix Size: "
        << N
        << " x "
        << N
        << "\n";

    std::cout
        << "Tile Size: "
        << TILE
        << " x "
        << TILE
        << "\n\n";


    // ==================================================
    // HOST MEMORY
    // ==================================================

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];

    float* h_CPU = new float[N * N];
    float* h_Naive = new float[N * N];
    float* h_Tiled = new float[N * N];


    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }


    // ==================================================
    // CPU BENCHMARK
    // ==================================================

    std::cout << "Running CPU matrix multiplication...\n";

    auto cpuStart =
        std::chrono::high_resolution_clock::now();

    cpuMatrixMul(
        h_A,
        h_B,
        h_CPU
    );

    auto cpuStop =
        std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(
            cpuStop - cpuStart
        ).count();


    // ==================================================
    // DEVICE MEMORY
    // ==================================================

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


    // ==================================================
    // CUDA CONFIGURATION
    // ==================================================

    dim3 threads(TILE, TILE);

    dim3 blocks(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );


    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // ==================================================
    // NAIVE GPU
    // ==================================================

    cudaEventRecord(start);

    naiveMatrixMul<<<blocks, threads>>>(
        d_A,
        d_B,
        d_C
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naiveTime = 0.0f;

    cudaEventElapsedTime(
        &naiveTime,
        start,
        stop
    );

    cudaMemcpy(
        h_Naive,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // ==================================================
    // TILED GPU
    // ==================================================

    cudaEventRecord(start);

    tiledMatrixMul<<<blocks, threads>>>(
        d_A,
        d_B,
        d_C
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tiledTime = 0.0f;

    cudaEventElapsedTime(
        &tiledTime,
        start,
        stop
    );

    cudaMemcpy(
        h_Tiled,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // ==================================================
    // RESULTS
    // ==================================================

    bool naiveCorrect =
        verify(h_CPU, h_Naive);

    bool tiledCorrect =
        verify(h_CPU, h_Tiled);


    std::cout
        << "\n--------------- RESULTS ---------------\n\n";

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "CPU Time       : "
        << cpuTime
        << " ms\n";

    std::cout
        << "Naive GPU Time : "
        << naiveTime
        << " ms\n";

    std::cout
        << "Tiled GPU Time : "
        << tiledTime
        << " ms\n";


    // ==================================================
    // SPEEDUPS
    // ==================================================

    double naiveSpeedup =
        cpuTime / naiveTime;

    double tiledSpeedup =
        cpuTime / tiledTime;

    double tilingSpeedup =
        naiveTime / tiledTime;


    std::cout
        << "\n----------------------------------------\n";

    std::cout
        << "CPU -> Naive GPU : "
        << naiveSpeedup
        << "x\n";

    std::cout
        << "CPU -> Tiled GPU : "
        << tiledSpeedup
        << "x\n";

    std::cout
        << "Naive -> Tiled   : "
        << tilingSpeedup
        << "x\n";


    // ==================================================
    // VERIFICATION
    // ==================================================

    std::cout
        << "\nNaive Verification: "
        << (naiveCorrect ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "Tiled Verification: "
        << (tiledCorrect ? "PASSED" : "FAILED")
        << "\n";


    // ==================================================
    // CLEANUP
    // ==================================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_A;
    delete[] h_B;
    delete[] h_CPU;
    delete[] h_Naive;
    delete[] h_Tiled;

    return 0;
}