#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#define CHECK_CUDA(call)                                              \
do {                                                                  \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        std::cerr << "CUDA Error: "                                  \
                  << cudaGetErrorString(err) << std::endl;   \
        return 1;                                                     \
    }                                                                 \
} while (0)

__global__ void vectorAdd(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(
        d_a,
        h_a,
        bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        d_b,
        h_b,
        bytes,
        cudaMemcpyHostToDevice
    ));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmup
    vectorAdd<<<blocks, threads>>>(
        d_a,
        d_b,
        d_c,
        N
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < 20; i++) {

        vectorAdd<<<blocks, threads>>>(
            d_a,
            d_b,
            d_c,
            N
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    
    // Updated: stop event passed correctly
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &ms,
            start,
            stop
        )
    );

    ms /= 20.0f;

    CHECK_CUDA(cudaMemcpy(
        h_c,
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool valid = true;

    for (int i = 0; i < N; i++) {

        if (h_c[i] != 3.0f) {
            valid = false;
            break;
        }
    }

    std::cout << "GPU-X Cloud Benchmark\n";
    std::cout << "=====================\n";
    std::cout << "Vector Size : " << N << "\n";
    std::cout << "Threads     : " << threads << "\n";
    std::cout << "Blocks      : " << blocks << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU Time    : " << ms << " ms\n";
    std::cout << "Verification: "
              << (valid ? "PASSED" : "FAILED")
              << "\n";

    cudaDeviceProp prop;

    CHECK_CUDA(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    std::cout << "GPU         : "
              << prop.name
              << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return valid ? 0 : 1;
}