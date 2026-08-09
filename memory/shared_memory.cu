#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

const int N = 16777216;
const int THREADS = 256;

// ==========================================
// GLOBAL MEMORY VERSION
// ==========================================

__global__ void globalMemoryAdd(
    const float* A,
    const float* B,
    float* C,
    int n
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        C[index] = A[index] + B[index];
    }
}


// ==========================================
// SHARED MEMORY VERSION
// ==========================================

__global__ void sharedMemoryAdd(
    const float* A,
    const float* B,
    float* C,
    int n
)
{
    __shared__ float sharedA[THREADS];
    __shared__ float sharedB[THREADS];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < n)
    {
        sharedA[tid] = A[index];
        sharedB[tid] = B[index];
    }

    __syncthreads();

    if (index < n)
    {
        C[index] = sharedA[tid] + sharedB[tid];
    }
}


// ==========================================
// MAIN
// ==========================================

int main()
{
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::cout << "========================================\n";
    std::cout << "      GPU-X SHARED MEMORY BENCHMARK\n";
    std::cout << "========================================\n\n";

    std::cout << "Elements: " << N << "\n";
    std::cout << "Threads per Block: " << THREADS << "\n";

    std::cout << "Data Size: "
              << bytes / (1024.0 * 1024.0)
              << " MB\n\n";


    // ==========================================
    // HOST MEMORY
    // ==========================================

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }


    // ==========================================
    // DEVICE MEMORY
    // ==========================================

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


    int blocks = (N + THREADS - 1) / THREADS;


    // ==========================================
    // CUDA EVENTS
    // ==========================================

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    float globalTime = 0.0f;
    float sharedTime = 0.0f;


    // ==========================================
    // GLOBAL MEMORY
    // ==========================================

    cudaEventRecord(start);

    globalMemoryAdd<<<blocks, THREADS>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(
        &globalTime,
        start,
        stop
    );


    // ==========================================
    // SHARED MEMORY
    // ==========================================

    cudaEventRecord(start);

    sharedMemoryAdd<<<blocks, THREADS>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(
        &sharedTime,
        start,
        stop
    );


    // ==========================================
    // COPY RESULT BACK
    // ==========================================

    cudaMemcpy(
        h_C,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // ==========================================
    // VERIFY
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

    std::cout << "Global Memory : "
              << globalTime
              << " ms\n";

    std::cout << "Shared Memory : "
              << sharedTime
              << " ms\n";

    std::cout << "\nResult Verification: "
              << (correct ? "PASSED" : "FAILED")
              << "\n";


    // ==========================================
    // SPEEDUP
    // ==========================================

    double speedup = globalTime / sharedTime;

    std::cout << "\nShared Memory Speedup: "
              << speedup
              << "x\n";


    // ==========================================
    // CLEANUP
    // ==========================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}