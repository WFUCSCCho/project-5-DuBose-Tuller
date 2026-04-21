#include <iostream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

// Max threads for each dimension of a 2D block
#ifndef T
#define T 32
#endif

using namespace std;

// Helper macro for error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

void printMatrix(float *A, int n) {
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            printf("%f\t", A[i*n + j]);
        }
        printf("\n");
    }
}

void writeMatrix(FILE *f, float *A, int n) {
    if (f == NULL) {
        printf("Error opening file");
        exit(4);
    }
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            fprintf(f, "%f\t", A[i*n + j]);
        }
        fprintf(f, "\n");
    }
}

// Separating init makes the generation kernel much faster
__global__ void setup_kernel(curandState_t* states, unsigned long seed, int n) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < n && col < n) {
        int idx = row * n + col;
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void generateRandomNumbers(curandState_t* states, float* numbers, int n) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < n && col < n) {
        int idx = row * n + col;
        // Use a local copy of the state for efficiency
        curandState_t localState = states[idx];
        numbers[idx] = curand_uniform(&localState);
        // Save the state back if you plan to call this kernel again
        states[idx] = localState;
    }
}

__global__ void multiplyMatrices(float *A, float *B, float *C, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col index

    if (i < n && j < n) {
        float c_ij = 0;

        for (int k=0; k<n; ++k) {
            c_ij += A[i*n + k] * B[k*n + j];
        }

        C[i*n + j] = c_ij;
    }
}


int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage" << argv[0] << " <n>" << endl;
        exit(2);
    }
    
    int n = strtol(argv[1], NULL, 10);
    printf("Performing random matmul with n = %d\n", n);

    // Initialize matrices
    int matrixSize = n*n*sizeof(float);
    float *A = (float*)malloc(matrixSize);
    float *B = (float*)malloc(matrixSize);
    float *C = (float*)malloc(matrixSize);
    if (A == NULL || B == NULL || C == NULL) {
        printf("malloc failed\n");
        exit(3);
    }

    float *dev_A, *dev_B, *dev_C;
    curandState_t *dev_states;

    CHECK_CUDA(cudaMalloc(&dev_A, matrixSize));
    CHECK_CUDA(cudaMalloc(&dev_B, matrixSize));
    CHECK_CUDA(cudaMalloc(&dev_C, matrixSize));
    CHECK_CUDA(cudaMalloc(&dev_states, n * n * sizeof(curandState_t)));

    dim3 dimGrid((n / T) + 1, (n / T) + 1, 1);
    dim3 dimBlock(T,T);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Fill input matrices with random values ON KERNEL
    // This is also considered the kernel's 'warm-up' for timing
    setup_kernel<<<dimGrid, dimBlock>>>(dev_states, time(NULL), n);
    generateRandomNumbers<<<dimGrid, dimBlock>>>(dev_states, dev_A, n);
    generateRandomNumbers<<<dimGrid, dimBlock>>>(dev_states, dev_B, n);

    CHECK_CUDA(cudaEventRecord(start)); // This happens on the GPU!
    CHECK_CUDA(cudaMemcpy(dev_C, C, matrixSize, cudaMemcpyHostToDevice));

    multiplyMatrices<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, n);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(C, dev_C, matrixSize, cudaMemcpyDeviceToHost));

    // Record the time for the *last* thread's cudaEventRecord(stop)
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Record the (matmul) kernel time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Write results to files
    FILE *product = fopen("product.dat", "w");
    writeMatrix(product, C, n);

    FILE *result = fopen("results.csv", "a");
    fprintf(result, "Naive,%d,%d,%f\n", n, T, milliseconds);

    // Free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    free(A);
    free(B);
    free(C);
}