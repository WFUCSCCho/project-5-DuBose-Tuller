/*
 * batched_ttv_omp.c  --  OpenMP batched tensor-times-vector (all six cases)
 *
 * For each case the two outermost loops are independent (they index output
 * elements) and are collapsed into a single parallel work-sharing loop.
 * The innermost loop is a scalar reduction with a thread-local accumulator,
 * so no atomic operations are needed.
 *
 * X and U are read from a binary file produced by ../generate/generate.
 *
 * Usage:  OMP_NUM_THREADS=<N> ./batched_ttv_omp <data_file>
 *
 * Appends one CSV line per case to results.csv:
 *   OpenMP-<N>,<case>,<I>,<J>,<K>,<seconds>
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static double elapsed_sec(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static void load_data(const char *path, int *I, int *J, int *K,
                      float **X, float **U)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fread(I, sizeof(int), 1, f);
    fread(J, sizeof(int), 1, f);
    fread(K, sizeof(int), 1, f);
    long IJK = (long)(*I) * (*J) * (*K);
    *X = (float *)malloc(IJK * sizeof(float));
    *U = (float *)malloc(IJK * sizeof(float));
    if (!*X || !*U) { fprintf(stderr, "malloc failed\n"); exit(2); }
    fread(*X, sizeof(float), IJK, f);
    fread(*U, sizeof(float), IJK, f);
    fclose(f);
}

/* ------------------------------------------------------------------
 * (m=1, n=3): Y[j + k*J] = sum_i  X[i + j*I + k*I*J] * U[i + k*I]
 * ------------------------------------------------------------------ */
static void ttv_m1_n3(const float *X, const float *U, float *Y,
                      int I, int J, int K)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < K; k++)
        for (int j = 0; j < J; j++) {
            float s = 0.0f;
            for (int i = 0; i < I; i++)
                s += X[i + j*I + k*I*J] * U[i + k*I];
            Y[j + k*J] = s;
        }
}

/* ------------------------------------------------------------------
 * (m=1, n=2): Y[k + j*K] = sum_i  X[i + j*I + k*I*J] * U[i + j*I]
 * ------------------------------------------------------------------ */
static void ttv_m1_n2(const float *X, const float *U, float *Y,
                      int I, int J, int K)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < J; j++)
        for (int k = 0; k < K; k++) {
            float s = 0.0f;
            for (int i = 0; i < I; i++)
                s += X[i + j*I + k*I*J] * U[i + j*I];
            Y[k + j*K] = s;
        }
}

/* ------------------------------------------------------------------
 * (m=2, n=3): Y[i + k*I] = sum_j  X[i + j*I + k*I*J] * U[j + k*J]
 * ------------------------------------------------------------------ */
static void ttv_m2_n3(const float *X, const float *U, float *Y,
                      int I, int J, int K)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < K; k++)
        for (int i = 0; i < I; i++) {
            float s = 0.0f;
            for (int j = 0; j < J; j++)
                s += X[i + j*I + k*I*J] * U[j + k*J];
            Y[i + k*I] = s;
        }
}

/* ------------------------------------------------------------------
 * (m=3, n=2): Y[i + j*I] = sum_k  X[i + j*I + k*I*J] * U[k + j*K]
 * ------------------------------------------------------------------ */
static void ttv_m3_n2(const float *X, const float *U, float *Y,
                      int I, int J, int K)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < J; j++)
        for (int i = 0; i < I; i++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += X[i + j*I + k*I*J] * U[k + j*K];
            Y[i + j*I] = s;
        }
}

/* ------------------------------------------------------------------
 * (m=2, n=1): Y[k + i*K] = sum_j  X[i + j*I + k*I*J] * U[j + i*J]
 * ------------------------------------------------------------------ */
static void ttv_m2_n1(const float *X, const float *U, float *Y,
                      int I, int J, int K)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < I; i++)
        for (int k = 0; k < K; k++) {
            float s = 0.0f;
            for (int j = 0; j < J; j++)
                s += X[i + j*I + k*I*J] * U[j + i*J];
            Y[k + i*K] = s;
        }
}

/* ------------------------------------------------------------------
 * (m=3, n=1): Y[j + i*J] = sum_k  X[i + j*I + k*I*J] * U[k + i*K]
 * ------------------------------------------------------------------ */
static void ttv_m3_n1(const float *X, const float *U, float *Y,
                      int I, int J, int K)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += X[i + j*I + k*I*J] * U[k + i*K];
            Y[j + i*J] = s;
        }
}

/* ================================================================== */

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <data_file>\n", argv[0]);
        return 1;
    }

    int I, J, K;
    float *X, *U;
    load_data(argv[1], &I, &J, &K, &X, &U);

    int nthreads;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }
    printf("Batched TTV (OpenMP x%d)  I=%d  J=%d  K=%d\n", nthreads, I, J, K);

    long IJK = (long)I * J * K;
    float *Y = (float *)malloc(IJK * sizeof(float));
    if (!Y) { fprintf(stderr, "malloc failed\n"); return 2; }

    struct timespec t0, t1;
    double sec;
    char label[32];
    snprintf(label, sizeof(label), "OpenMP-%d", nthreads);

    FILE *csv = fopen("results.csv", "a");
    if (!csv) { fprintf(stderr, "Cannot open results.csv\n"); return 3; }

    /* ---------- (m=1, n=3) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m1_n3(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=1 n=3  out %d x %d : %.6f s\n", J, K, sec);
    fprintf(csv, "%s,m1n3,%d,%d,%d,%.6f\n", label, I, J, K, sec);

    /* ---------- (m=1, n=2) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m1_n2(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=1 n=2  out %d x %d : %.6f s\n", K, J, sec);
    fprintf(csv, "%s,m1n2,%d,%d,%d,%.6f\n", label, I, J, K, sec);

    /* ---------- (m=2, n=3) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m2_n3(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=2 n=3  out %d x %d : %.6f s\n", I, K, sec);
    fprintf(csv, "%s,m2n3,%d,%d,%d,%.6f\n", label, I, J, K, sec);

    /* ---------- (m=3, n=2) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m3_n2(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=3 n=2  out %d x %d : %.6f s\n", I, J, sec);
    fprintf(csv, "%s,m3n2,%d,%d,%d,%.6f\n", label, I, J, K, sec);

    /* ---------- (m=2, n=1) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m2_n1(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=2 n=1  out %d x %d : %.6f s\n", K, I, sec);
    fprintf(csv, "%s,m2n1,%d,%d,%d,%.6f\n", label, I, J, K, sec);

    /* ---------- (m=3, n=1) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m3_n1(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=3 n=1  out %d x %d : %.6f s\n", J, I, sec);
    fprintf(csv, "%s,m3n1,%d,%d,%d,%.6f\n", label, I, J, K, sec);

    fclose(csv);
    free(X); free(U); free(Y);
    return 0;
}
