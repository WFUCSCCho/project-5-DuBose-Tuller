/*
 * generate.c -- Write a reproducible random tensor X and vector buffer U
 *               to a binary file consumed by all benchmark variants.
 *
 * File format (native endian, same machine assumed):
 *   int32   I, J, K
 *   float   X[I*J*K]             column-major: X[i + j*I + k*I*J]
 *   float   U[max(IJ, IK, JK)]   sufficient for all six TTV cases
 *
 * Usage: ./generate <I> <J> <K> <output_path>
 */
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <I> <J> <K> <output_path>\n", argv[0]);
        return 1;
    }
    int I = atoi(argv[1]);
    int J = atoi(argv[2]);
    int K = atoi(argv[3]);
    const char *path = argv[4];

    long IJK  = (long)I * J * K;
    long IJ   = (long)I * J;
    long IK   = (long)I * K;
    long JK   = (long)J * K;
    long Ulen = IJ > IK ? IJ : IK;
    if (JK > Ulen) Ulen = JK;

    float *X = (float *)malloc(IJK  * sizeof(float));
    float *U = (float *)malloc(Ulen * sizeof(float));
    if (!X || !U) { fprintf(stderr, "malloc failed\n"); return 2; }

    srand(42);
    for (long n = 0; n < IJK;  n++) X[n] = (float)rand() / RAND_MAX;
    for (long n = 0; n < Ulen; n++) U[n] = (float)rand() / RAND_MAX;

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", path); return 3; }
    fwrite(&I, sizeof(int),   1,    f);
    fwrite(&J, sizeof(int),   1,    f);
    fwrite(&K, sizeof(int),   1,    f);
    fwrite(X,  sizeof(float), IJK,  f);
    fwrite(U,  sizeof(float), Ulen, f);
    fclose(f);

    printf("Wrote %s  I=%d J=%d K=%d  X=%.1f MB  U=%.1f MB\n",
           path, I, J, K,
           (double)(IJK  * sizeof(float)) / (1 << 20),
           (double)(Ulen * sizeof(float)) / (1 << 20));
    free(X); free(U);
    return 0;
}
