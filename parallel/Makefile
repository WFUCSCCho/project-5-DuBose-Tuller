NVCC    = nvcc
CC      = gcc
CFLAGS  = -fopenmp -Wall

all: batched_ttv batched_ttv_omp

batched_ttv: batched_ttv.cu
	$(NVCC) -o batched_ttv batched_ttv.cu

batched_ttv_omp: batched_ttv_omp.c
	$(CC) $(CFLAGS) -o batched_ttv_omp batched_ttv_omp.c

clean:
	rm -f batched_ttv batched_ttv_omp
