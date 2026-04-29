#!/bin/bash                                                                                                               
#SBATCH --job-name "bttv_bench"
#SBATCH --output=./logs/bench-%j.out                                                                                        
#SBATCH --error=./logs/bench-%j.err
#SBATCH --partition=gpu                                                                                                     
#SBATCH --nodes=1                                                                                                         
#SBATCH --gres=gpu:1                                                                                                        
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G                                                                                                           
#SBATCH --time=01:00:00                                                                                                   

cd /deac/csc/classes/csc347/tullwd25/project-5-DuBose-Tuller

DATA=data/1000_1000_1000.bin    

cd parallel/
make
cd ../serial/
make
cd ..                                                                                     

# Serial baseline                                                                                                           
./serial/batched_ttv $DATA                                                                                                

# OpenMP scaling sweep                                                                                                      
for N in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$N                                                                                               
    ./parallel/batched_ttv_omp $DATA                                                                                      
done                                                                                                                        

# GPU                                                                                                                       
./parallel/batched_ttv $DATA
