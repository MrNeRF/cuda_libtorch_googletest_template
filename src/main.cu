#include <cuda_runtime.h>
#include "error_checking.cuh"
#include <iostream>

int main(int argc, char** argv) {
    float* d_input;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, 784 * sizeof(float))); // Allocate device memory for input
    cudaFree(d_input);
    std::cout << "Hello World!\n";
    return 0;
}