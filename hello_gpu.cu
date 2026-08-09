#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello()
{
    printf("Hello from RTX 2050!\n");
}

int main()
{
    hello<<<1, 1>>>();

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA kernel executed successfully!\n");

    return 0;
}