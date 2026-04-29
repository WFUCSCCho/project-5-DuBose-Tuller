#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

static double elapsed_sec(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static void load_data(const char *path, int *I, int *J, int *K,
                      float **X, float **U)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fread(I, sizeof(int), 1, f);
    fread(J, sizeof(int), 1, f);
    fread(K, sizeof(int), 1, f);
    long IJK  = (long)(*I) * (*J) * (*K);
    long IJ   = (long)(*I) * (*J);
    long IK   = (long)(*I) * (*K);
    long JK   = (long)(*J) * (*K);
    long Ulen = IJ > IK ? IJ : IK;
    if (JK > Ulen) Ulen = JK;
    *X = (float *)malloc(IJK  * sizeof(float));
    *U = (float *)malloc(Ulen * sizeof(float));
    if (*X == NULL || *U == NULL) { fprintf(stderr, "malloc failed\n"); exit(2); }
    fread(*X, sizeof(float), IJK,  f);
    fread(*U, sizeof(float), Ulen, f);
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
 * Case (m=1, n=3): contract mode 1 (size I), batch over mode 3 (K)
 * Y: J x K   ->  Y[j + k*J] = sum_i  X[i + j*I + k*I*J] * U[i + k*I]
 * threads: x=j, y=local_k   (k is free in output)
 * ------------------------------------------------------------------ */
__global__ void bttv_m_1_n_3(float *X, float *U, float *Y, int I, int J, int K, int chunk_k, int k_offset) {
    int j       = threadIdx.x + blockIdx.x * blockDim.x;
    int local_k = threadIdx.y + blockIdx.y * blockDim.y;
    if (j < J && local_k < chunk_k) {
        int k = k_offset + local_k;
        float sum = 0;
        for (int i = 0; i < I; ++i) {
            sum += X[i + j*I + (long)local_k*I*J] * U[i + k*I];
        }
        Y[j + k*J] = sum;
    }
}

/* ------------------------------------------------------------------
 * Case (m=1, n=2): contract mode 1 (size I), batch over mode 2 (J)
 * Y: K x J   ->  Y[k + j*K] = sum_i  X[i + j*I + k*I*J] * U[i + j*I]
 * threads: x=j, y=local_k   (k is free in output)
 * ------------------------------------------------------------------ */
__global__ void bttv_m_1_n_2(float *X, float *U, float *Y, int I, int J, int K, int chunk_k, int k_offset) {
    int j       = threadIdx.x + blockIdx.x * blockDim.x;
    int local_k = threadIdx.y + blockIdx.y * blockDim.y;
    if (j < J && local_k < chunk_k) {
        int k = k_offset + local_k;
        float sum = 0;
        for (int i = 0; i < I; ++i) {
            sum += X[i + j*I + (long)local_k*I*J] * U[i + j*I];
        }
        Y[k + j*K] = sum;
    }
}

/* ------------------------------------------------------------------
 * Case (m=2, n=3): contract mode 2 (size J), batch over mode 3 (K)
 * Y: I x K   ->  Y[i + k*I] = sum_j  X[i + j*I + k*I*J] * U[j + k*J]
 * threads: x=i, y=local_k   (k is free in output)
 * ------------------------------------------------------------------ */
__global__ void bttv_m_2_n_3(float *X, float *U, float *Y, int I, int J, int K, int chunk_k, int k_offset) {
    int i       = threadIdx.x + blockIdx.x * blockDim.x;
    int local_k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < I && local_k < chunk_k) {
        int k = k_offset + local_k;
        float sum = 0;
        for (int j = 0; j < J; ++j) {
            sum += X[i + j*I + (long)local_k*I*J] * U[j + k*J];
        }
        Y[i + k*I] = sum;
    }
}

/* ------------------------------------------------------------------
 * Case (m=3, n=2): contract mode 3 (size K), batch over mode 2 (J)
 * Y: I x J   ->  Y[i + j*I] = sum_k  X[i + j*I + k*I*J] * U[k + j*K]
 * threads: x=i, y=j   (k contracted; accumulates across chunks into Y)
 * ------------------------------------------------------------------ */
__global__ void bttv_m_3_n_2(float *X, float *U, float *Y, int I, int J, int K, int chunk_k, int k_offset) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < I && j < J) {
        float sum = 0;
        for (int lk = 0; lk < chunk_k; ++lk) {
            sum += X[i + j*I + (long)lk*I*J] * U[(k_offset + lk) + j*K];
        }
        Y[i + j*I] += sum;
    }
}

/* ------------------------------------------------------------------
 * Case (m=2, n=1): contract mode 2 (size J), batch over mode 1 (I)
 * Y: K x I   ->  Y[k + i*K] = sum_j  X[i + j*I + k*I*J] * U[j + i*J]
 * U layout: J x I  (column i holds the vector for batch element i)
 * ------------------------------------------------------------------ */
__global__ void bttv_m_2_n_1(float *X, float *U, float *Y, int I, int J, int K, int chunk_k, int k_offset) {
    // threadIdx.x -> i: consecutive threads read X[i + j*I + k*I*J] with stride 1
    int i       = threadIdx.x + blockIdx.x * blockDim.x;
    int local_k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < I && local_k < chunk_k) {
        float sum = 0;
        for (int j = 0; j < J; ++j) {
            sum += X[i + j*I + (long)local_k*I*J] * U[j + i*J];
        }
        Y[(k_offset + local_k) + i*K] = sum;
    }
}

/* ------------------------------------------------------------------
 * Case (m=3, n=1): contract mode 3 (size K), batch over mode 1 (I)
 * Y: J x I   ->  Y[j + i*J] = sum_k  X[i + j*I + k*I*J] * U[k + i*K]
 * threads: x=i, y=j   (k contracted; accumulates across chunks into Y)
 * ------------------------------------------------------------------ */
__global__ void bttv_m_3_n_1(float *X, float *U, float *Y, int I, int J, int K, int chunk_k, int k_offset) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < I && j < J) {
        float sum = 0;
        for (int lk = 0; lk < chunk_k; ++lk) {
            sum += X[i + j*I + (long)lk*I*J] * U[(k_offset + lk) + i*K];
        }
        Y[j + i*J] += sum;
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

    long IJ = (long)I*J, IK = (long)I*K, JK = (long)J*K;
    long Ulen = IJ > IK ? IJ : IK;
    if (JK > Ulen) Ulen = JK;
    long Ymax = Ulen; /* max output size = max(IJ, IK, JK) */

    // Initialize arrays
    float *Y = (float*)malloc(Ymax * sizeof(float));
    if (Y == NULL) {
        printf("malloc failed\n");
        exit(3);
    }

    // How many k-slices (each I*J floats) fit in free VRAM?
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    size_t headroom = 256UL << 20;
    size_t usable = free_mem > headroom ? free_mem - headroom : free_mem / 2;
    int chunk_k = (int)(usable / (IJ * sizeof(float)));
    if (chunk_k < 1) chunk_k = 1;
    if (chunk_k > K) chunk_k = K;
    printf("  GPU %.1f/%.1f GB free, chunk_k=%d (of K=%d)\n",
           free_mem/1e9, total_mem/1e9, chunk_k, K);

    float *dev_X, *dev_U, *dev_Y;
    CHECK_CUDA(cudaMalloc(&dev_X, IJ * chunk_k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_U, Ulen * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_Y, Ymax * sizeof(float)));

    // Copy U once; it is constant across all six cases
    CHECK_CUDA(cudaMemcpy(dev_U, U, Ulen * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(T, T);
    struct timespec t0, t1;
    double sec;

    FILE *result = fopen("results.csv", "a");
    if (!result) { printf("Cannot open results.csv\n"); exit(4); }

    /* ---------- (m=1, n=3) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k_off = 0; k_off < K; k_off += chunk_k) {
        int this_k = (k_off + chunk_k <= K) ? chunk_k : K - k_off;
        CHECK_CUDA(cudaMemcpy(dev_X, X + (long)k_off*IJ, IJ*this_k*sizeof(float), cudaMemcpyHostToDevice));
        dim3 dimGrid((J+T-1)/T, (this_k+T-1)/T);
        bttv_m_1_n_3<<<dimGrid, dimBlock>>>(dev_X, dev_U, dev_Y, I, J, K, this_k, k_off);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, dev_Y, JK*sizeof(float), cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("Kernel execution time: %f ms\n", sec * 1000.0);
    fprintf(result, "GPU,m1n3,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=1, n=2) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k_off = 0; k_off < K; k_off += chunk_k) {
        int this_k = (k_off + chunk_k <= K) ? chunk_k : K - k_off;
        CHECK_CUDA(cudaMemcpy(dev_X, X + (long)k_off*IJ, IJ*this_k*sizeof(float), cudaMemcpyHostToDevice));
        dim3 dimGrid((J+T-1)/T, (this_k+T-1)/T);
        bttv_m_1_n_2<<<dimGrid, dimBlock>>>(dev_X, dev_U, dev_Y, I, J, K, this_k, k_off);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, dev_Y, JK*sizeof(float), cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("Kernel execution time: %f ms\n", sec * 1000.0);
    fprintf(result, "GPU,m1n2,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=2, n=3) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k_off = 0; k_off < K; k_off += chunk_k) {
        int this_k = (k_off + chunk_k <= K) ? chunk_k : K - k_off;
        CHECK_CUDA(cudaMemcpy(dev_X, X + (long)k_off*IJ, IJ*this_k*sizeof(float), cudaMemcpyHostToDevice));
        dim3 dimGrid((I+T-1)/T, (this_k+T-1)/T);
        bttv_m_2_n_3<<<dimGrid, dimBlock>>>(dev_X, dev_U, dev_Y, I, J, K, this_k, k_off);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, dev_Y, IK*sizeof(float), cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("Kernel execution time: %f ms\n", sec * 1000.0);
    fprintf(result, "GPU,m2n3,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=3, n=2): k contracted, zero Y before accumulating ---------- */
    CHECK_CUDA(cudaMemset(dev_Y, 0, IJ*sizeof(float)));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k_off = 0; k_off < K; k_off += chunk_k) {
        int this_k = (k_off + chunk_k <= K) ? chunk_k : K - k_off;
        CHECK_CUDA(cudaMemcpy(dev_X, X + (long)k_off*IJ, IJ*this_k*sizeof(float), cudaMemcpyHostToDevice));
        dim3 dimGrid((I+T-1)/T, (J+T-1)/T);
        bttv_m_3_n_2<<<dimGrid, dimBlock>>>(dev_X, dev_U, dev_Y, I, J, K, this_k, k_off);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, dev_Y, IJ*sizeof(float), cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("Kernel execution time: %f ms\n", sec * 1000.0);
    fprintf(result, "GPU,m3n2,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=2, n=1) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k_off = 0; k_off < K; k_off += chunk_k) {
        int this_k = (k_off + chunk_k <= K) ? chunk_k : K - k_off;
        CHECK_CUDA(cudaMemcpy(dev_X, X + (long)k_off*IJ, IJ*this_k*sizeof(float), cudaMemcpyHostToDevice));
        dim3 dimGrid((I+T-1)/T, (this_k+T-1)/T);
        bttv_m_2_n_1<<<dimGrid, dimBlock>>>(dev_X, dev_U, dev_Y, I, J, K, this_k, k_off);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, dev_Y, IK*sizeof(float), cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("Kernel execution time: %f ms\n", sec * 1000.0);
    fprintf(result, "GPU,m2n1,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=3, n=1): k contracted, zero Y before accumulating ---------- */
    CHECK_CUDA(cudaMemset(dev_Y, 0, IJ*sizeof(float)));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k_off = 0; k_off < K; k_off += chunk_k) {
        int this_k = (k_off + chunk_k <= K) ? chunk_k : K - k_off;
        CHECK_CUDA(cudaMemcpy(dev_X, X + (long)k_off*IJ, IJ*this_k*sizeof(float), cudaMemcpyHostToDevice));
        dim3 dimGrid((I+T-1)/T, (J+T-1)/T);
        bttv_m_3_n_1<<<dimGrid, dimBlock>>>(dev_X, dev_U, dev_Y, I, J, K, this_k, k_off);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, dev_Y, IJ*sizeof(float), cudaMemcpyDeviceToHost));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("Kernel execution time: %f ms\n", sec * 1000.0);
    fprintf(result, "GPU,m3n1,%d,%d,%d,%.6f\n", I, J, K, sec);

    fclose(result);

    // Free
    cudaFree(dev_X);
    cudaFree(dev_U);
    cudaFree(dev_Y);
    free(X);
    free(U);
    free(Y);
}
