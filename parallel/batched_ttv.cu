#include <iostream>
#include <cuda.h>

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

static void load_data(const char *path, int *I, int *J, int *K,
                      float **X, float **U)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fread(I, sizeof(int), 1, f);
    fread(J, sizeof(int), 1, f);
    fread(K, sizeof(int), 1, f);
    long IJK = (long)(*I) * (*J) * (*K);
    *X = (float *)malloc(IJK * sizeof(float));
    *U = (float *)malloc(IJK * sizeof(float));
    if (*X == NULL || *U == NULL) { fprintf(stderr, "malloc failed\n"); exit(2); }
    fread(*X, sizeof(float), IJK, f);
    fread(*U, sizeof(float), IJK, f);
    fclose(f);
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

/* ------------------------------------------------------------------
 * Case (m=2, n=1): contract mode 2 (size J), batch over mode 1 (I)
 * Y: K x I   ->  Y[k + i*K] = sum_j  X[i + j*I + k*I*J] * U[j + i*J]
 * U layout: J x I  (column i holds the vector for batch element i)
 * ------------------------------------------------------------------ */
__global__ void bttv_m_2_n_1(float *X, float *U, float *Y, int I, int J, int K) {
    // threadIdx.x -> i: consecutive threads read X[i + j*I + k*I*J] with stride 1
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
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <data_file>" << endl;
        exit(2);
    }

    // Load data from binary file
    int I, J, K;
    float *X, *U;
    load_data(argv[1], &I, &J, &K, &X, &U);
    printf("Performing bttv with I=%d J=%d K=%d\n", I, J, K);

    // Initialize arrays
    float *Y = (float*)malloc((long)K*I*sizeof(float));
    if (Y == NULL) {
        printf("malloc failed\n");
        exit(3);
    }

    float *dev_X, *dev_U, *dev_Y;

    CHECK_CUDA(cudaMalloc(&dev_X, (long)I*J*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_U, (long)J*I*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_Y, (long)K*I*sizeof(float)));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(dev_X, X, (long)I*J*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_U, U, (long)J*I*sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimGrid((I / T) + 1, (K / T) + 1);
    dim3 dimBlock(T,T);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start)); // This happens on the GPU!

    bttv_m_2_n_1<<<dimGrid, dimBlock>>>(dev_X, dev_U, dev_Y, I, J, K);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, dev_Y, K*I*sizeof(float), cudaMemcpyDeviceToHost));

    // Record the time for the *last* thread's cudaEventRecord(stop)
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Record the kernel time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Write results to files
    FILE *product = fopen("product.dat", "w");
    writeMatrix(product, Y, K, I);
    fclose(product);

    FILE *result = fopen("results.csv", "a");
    fprintf(result, "GPU,m2n1,%d,%d,%d,%.6f\n", I, J, K, milliseconds / 1000.0f);
    fclose(result);

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