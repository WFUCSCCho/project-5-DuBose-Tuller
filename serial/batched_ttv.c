/*
 * batched_ttv.c  --  Serial (naive) batched tensor-times-vector
 *
 * Derived from: ~/Documents/MATLAB/tensors/cp-newton/matlab/batched_ttv.c
 * Original used the MATLAB MEX interface and BLAS (dgemv/ddot).
 * This version removes both dependencies and replaces them with plain
 * nested loops, following the style of the course MatrixMult.c serial
 * baseline.
 *
 * Problem: given a 3rd-order tensor X of size I x J x K (stored in
 * column-major order, i.e. X[i + j*I + k*I*J]) and a batch of vectors
 * packed into U, compute the mode-m tensor-times-vector product,
 * batched over mode n, for all six (m, n) pairings.
 *
 * Usage:  ./batched_ttv <I> <J> <K>
 *
 * Output dimensions for each case:
 *   (m=1,n=3) -> Y: J x K
 *   (m=1,n=2) -> Y: K x J
 *   (m=2,n=3) -> Y: I x K
 *   (m=3,n=2) -> Y: I x J
 *   (m=2,n=1) -> Y: K x I
 *   (m=3,n=1) -> Y: J x I
 *
 * Timing for each case is appended to results.csv.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Return elapsed wall-clock seconds between two timespec values. */
static double elapsed_sec(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}


/* ------------------------------------------------------------------
 * Case (m=1, n=3): contract mode 1 (size I), batch over mode 3 (K)
 * Y: J x K   ->  Y[j + k*J] = sum_i  X[i + j*I + k*I*J] * U[i + k*I]
 * U layout: I x K  (column k holds the vector for batch element k)
 * ------------------------------------------------------------------ */
void ttv_m1_n3(const double *X, const double *U, double *Y,
               int I, int J, int K)
{
    for (int k = 0; k < K; k++)
        for (int j = 0; j < J; j++) {
            double s = 0.0;
            for (int i = 0; i < I; i++)
                s += X[i + j*I + k*I*J] * U[i + k*I];
            Y[j + k*J] = s;
        }
}

/* ------------------------------------------------------------------
 * Case (m=1, n=2): contract mode 1 (size I), batch over mode 2 (J)
 * Y: K x J   ->  Y[k + j*K] = sum_i  X[i + j*I + k*I*J] * U[i + j*I]
 * U layout: I x J  (column j holds the vector for batch element j)
 * ------------------------------------------------------------------ */
void ttv_m1_n2(const double *X, const double *U, double *Y,
               int I, int J, int K)
{
    for (int j = 0; j < J; j++)
        for (int k = 0; k < K; k++) {
            double s = 0.0;
            for (int i = 0; i < I; i++)
                s += X[i + j*I + k*I*J] * U[i + j*I];
            Y[k + j*K] = s;
        }
}

/* ------------------------------------------------------------------
 * Case (m=2, n=3): contract mode 2 (size J), batch over mode 3 (K)
 * Y: I x K   ->  Y[i + k*I] = sum_j  X[i + j*I + k*I*J] * U[j + k*J]
 * U layout: J x K  (column k holds the vector for batch element k)
 * ------------------------------------------------------------------ */
void ttv_m2_n3(const double *X, const double *U, double *Y,
               int I, int J, int K)
{
    for (int k = 0; k < K; k++)
        for (int i = 0; i < I; i++) {
            double s = 0.0;
            for (int j = 0; j < J; j++)
                s += X[i + j*I + k*I*J] * U[j + k*J];
            Y[i + k*I] = s;
        }
}

/* ------------------------------------------------------------------
 * Case (m=3, n=2): contract mode 3 (size K), batch over mode 2 (J)
 * Y: I x J   ->  Y[i + j*I] = sum_k  X[i + j*I + k*I*J] * U[k + j*K]
 * U layout: K x J  (column j holds the vector for batch element j)
 * ------------------------------------------------------------------ */
void ttv_m3_n2(const double *X, const double *U, double *Y,
               int I, int J, int K)
{
    for (int j = 0; j < J; j++)
        for (int i = 0; i < I; i++) {
            double s = 0.0;
            for (int k = 0; k < K; k++)
                s += X[i + j*I + k*I*J] * U[k + j*K];
            Y[i + j*I] = s;
        }
}

/* ------------------------------------------------------------------
 * Case (m=2, n=1): contract mode 2 (size J), batch over mode 1 (I)
 * Y: K x I   ->  Y[k + i*K] = sum_j  X[i + j*I + k*I*J] * U[j + i*J]
 * U layout: J x I  (column i holds the vector for batch element i)
 * ------------------------------------------------------------------ */
void ttv_m2_n1(const double *X, const double *U, double *Y,
               int I, int J, int K)
{
    for (int i = 0; i < I; i++)
        for (int k = 0; k < K; k++) {
            double s = 0.0;
            for (int j = 0; j < J; j++)
                s += X[i + j*I + k*I*J] * U[j + i*J];
            Y[k + i*K] = s;
        }
}

/* ------------------------------------------------------------------
 * Case (m=3, n=1): contract mode 3 (size K), batch over mode 1 (I)
 * Y: J x I   ->  Y[j + i*J] = sum_k  X[i + j*I + k*I*J] * U[k + i*K]
 * U layout: K x I  (column i holds the vector for batch element i)
 * ------------------------------------------------------------------ */
void ttv_m3_n1(const double *X, const double *U, double *Y,
               int I, int J, int K)
{
    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++) {
            double s = 0.0;
            for (int k = 0; k < K; k++)
                s += X[i + j*I + k*I*J] * U[k + i*K];
            Y[j + i*J] = s;
        }
}

/* ================================================================== */

int main(int argc, char *argv[])
{
    if (argc != 4) {
        fprintf(stderr, "Usage: ./batched_ttv <I> <J> <K>\n");
        return 1;
    }

    int I = atoi(argv[1]);
    int J = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("Batched TTV  I=%d  J=%d  K=%d\n", I, J, K);

    long IJK = (long)I * J * K;

    /* X: the I x J x K tensor (column-major).
     * U: batch of vectors.  Allocating IJK elements is sufficient for
     *    every case since each U layout is a product of two dimensions,
     *    which is always <= I*J*K for I,J,K >= 1.
     * Y: output; same argument -- largest output is also a two-dim product. */
    double *X = (double *)malloc(IJK * sizeof(double));
    double *U = (double *)malloc(IJK * sizeof(double));
    double *Y = (double *)malloc(IJK * sizeof(double));
    if (!X || !U || !Y) {
        fprintf(stderr, "malloc failed\n");
        return 2;
    }

    srand(42);
    for (long idx = 0; idx < IJK; idx++) {
        X[idx] = (double)rand() / RAND_MAX;
        U[idx] = (double)rand() / RAND_MAX;
    }

    struct timespec t0, t1;
    double sec;

    FILE *csv = fopen("results.csv", "a");
    if (!csv) { fprintf(stderr, "Cannot open results.csv\n"); return 3; }

    /* ---------- (m=1, n=3) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m1_n3(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=1 n=3  out %d x %d : %.6f s\n", J, K, sec);
    fprintf(csv, "Serial,m1n3,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=1, n=2) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m1_n2(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=1 n=2  out %d x %d : %.6f s\n", K, J, sec);
    fprintf(csv, "Serial,m1n2,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=2, n=3) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m2_n3(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=2 n=3  out %d x %d : %.6f s\n", I, K, sec);
    fprintf(csv, "Serial,m2n3,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=3, n=2) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m3_n2(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=3 n=2  out %d x %d : %.6f s\n", I, J, sec);
    fprintf(csv, "Serial,m3n2,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=2, n=1) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m2_n1(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=2 n=1  out %d x %d : %.6f s\n", K, I, sec);
    fprintf(csv, "Serial,m2n1,%d,%d,%d,%.6f\n", I, J, K, sec);

    /* ---------- (m=3, n=1) ---------- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ttv_m3_n1(X, U, Y, I, J, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = elapsed_sec(t0, t1);
    printf("  m=3 n=1  out %d x %d : %.6f s\n", J, I, sec);
    fprintf(csv, "Serial,m3n1,%d,%d,%d,%.6f\n", I, J, K, sec);

    fclose(csv);
    free(X);
    free(U);
    free(Y);
    return 0;
}
