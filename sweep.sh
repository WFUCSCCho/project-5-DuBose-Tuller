#!/bin/bash
#SBATCH --job-name "bttv_bench"
#SBATCH --output=./logs/bench-%j.out
#SBATCH --error=./logs/bench-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G 
#SBATCH --time=02:00:00

ROOT=/deac/csc/classes/csc347/tullwd25/project-5-DuBose-Tuller
cd $ROOT

cd generate && make -s && cd ..
cd parallel  && make -s && cd ..
cd serial    && make -s && cd ..

mkdir -p data logs

# ── Generate data files ───────────────────────────────────────────────
# If 40 GB is not enough for generate at n=2000, pre-generate on a login node:
#   ./generate/generate 2000 2000 2000 data/2000_2000_2000.bin
for N in 500 750 1000 1250 1500 1750 2000; do
    BIN=data/${N}_${N}_${N}.bin
    [ -f "$BIN" ] || ./generate/generate $N $N $N "$BIN"
done

# ── Fig 3: size sweep — GPU and OMP-32 at all sizes ──────────────────
for SZ in 500 750 1000 1250 1500 1750 2000; do
    DATA=data/${SZ}_${SZ}_${SZ}.bin
    ./parallel/batched_ttv "$DATA"
    OMP_NUM_THREADS=32 ./parallel/batched_ttv_omp "$DATA"
done

# Serial at small sizes only: Y is over-allocated to I*J*K floats,
# so n=1000 costs ~8 GB (X + Y); n=1250 would need ~16 GB.
for SZ in 500 750 1000; do
    ./serial/batched_ttv "data/${SZ}_${SZ}_${SZ}.bin"
done

# ── Fig 2: thread scaling at n=1000 ──────────────────────────────────
# OMP-32 already captured above; run remaining thread counts here.
DATA=data/1000_1000_1000.bin
for NP in 1 2 4 8 16; do
    OMP_NUM_THREADS=$NP ./parallel/batched_ttv_omp "$DATA"
done
