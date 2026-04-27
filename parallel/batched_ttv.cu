#include <iostream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>
#include <curand_kernel.h>

using namespace std;
#define T 32

// Helper macro for error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

void printMatrix(float *A, int m, int n) {
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            printf("%f\t", A[i*n + j]);
        }
        printf("\n");
    }
}

void writeMatrix(FILE *f, float *A, int m, int n) {
    if (f == NULL) {
        printf("Error opening file");
        exit(4);
    }
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            fprintf(f, "%f\t", A[i*n + j]);
        }
        fprintf(f, "\n");
    }
}

// Separating init makes the generation kernel much faster
__global__ void setup_kernel(curandState_t* states, unsigned long seed, int I, int J, int K) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int slc = threadIdx.z + blockIdx.z * blockDim.z;
    if (row < I && col < J && slc < K) {
        int idx = row + col * I + slc * I * J;
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void generateRandomNumbers(curandState_t* states, float* numbers, int I, int J, int K) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int slc = threadIdx.z + blockIdx.z * blockDim.z;

    if (row < I && col < J && slc < K) {
        int idx = row + col * I + slc * I * J;
        // Use a local copy of the state for efficiency
        curandState_t localState = states[idx];
        numbers[idx] = curand_uniform(&localState);
        // Save the state back if you plan to call this kernel again
        states[idx] = localState;
    }
}

/* ------------------------------------------------------------------
 * Case (m=2, n=1): contract mode 2 (size J), batch over mode 1 (I)
 * Y: K x I   ->  Y[k + i*K] = sum_j  X[i + j*I + k*I*J] * U[j + i*J]
 * U layout: J x I  (column i holds the vector for batch element i)
 * ------------------------------------------------------------------ */
__global__ void bttv_m_2_n_1(float *X, float *U, float *Y, int I, int J, int K) {
    // threadIdx.x → i: consecutive threads read X[i + j*I + k*I*J] with stride 1
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < I && k < K) {
        float sum = 0;
        for (int j = 0; j < J; ++j) {
            sum += X[i + j*I + k*I*J] * U[j + i*J];
        }
        Y[k + i*K] = sum;
    }
}


int main(int argc, char **argv) {
    if (argc != 4) {
        cerr << "Usage" << argv[0] << " <I>, <J>, <K>" << endl;
        exit(2);
    }
    
    int I = strtol(argv[1], NULL, 10);
    int J = strtol(argv[2], NULL, 10);
    int K = strtol(argv[3], NULL, 10);
    printf("Performing random bttv with I=%d J=%d K=%d\n", I, J, K);

    // Initialize arrays
    float *X = (float*)malloc(I*J*K*sizeof(float));
    float *U = (float*)malloc(J*I*sizeof(float));
    float *Y = (float*)malloc(K*I*sizeof(float));
    if (X == NULL || U == NULL || Y == NULL) {
        printf("malloc failed\n");
        exit(3);
    }

    float *dev_X, *dev_U, *dev_Y;
    curandState_t *dev_states;

    CHECK_CUDA(cudaMalloc(&dev_X, I*J*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_U, J*I*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_Y, K*I*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_states, (long) I * J * K * sizeof(curandState_t)));

    dim3 dimGrid3((J / T) + 1, (I / T) + 1, (K / T) + 1); // For X init
    dim3 dimGrid((I / T) + 1, (K / T) + 1);
    dim3 dimBlock(T,T);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Fill input matrices with random values ON KERNEL
    // This is also considered the kernel's 'warm-up' for timing
    setup_kernel<<<dimGrid3, dimBlock>>>(dev_states, time(NULL), I, J, K);
    generateRandomNumbers<<<dimGrid3, dimBlock>>>(dev_states, dev_X, I, J, K);
    generateRandomNumbers<<<dimGrid3, dimBlock>>>(dev_states, dev_U, J, I, 1);

    CHECK_CUDA(cudaEventRecord(start)); // This happens on the GPU!

    bttv_m_2_n_1<<<dimGrid, dimBlock>>>(dev_X, dev_U, dev_Y, I, J, K);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, dev_Y, K*I*sizeof(float), cudaMemcpyDeviceToHost));

    // Record the time for the *last* thread's cudaEventRecord(stop)
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Record the (matmul) kernel time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Write results to files
    FILE *product = fopen("product.dat", "w");
    writeMatrix(product, Y, K, I);

    FILE *result = fopen("results.csv", "a");
    fprintf(result, "Naive,m2n1,%d,%d,%d,%.6f\n", I, J, K, milliseconds / 1000.0f);

    // Free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_X);
    cudaFree(dev_U);
    cudaFree(dev_Y);
    free(X);
    free(U);
    free(Y);
}