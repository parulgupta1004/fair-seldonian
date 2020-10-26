#!/bin/sh
#SBATCH -J base
#SBATCH --mem=40000 --partition=longq --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --output=logs-base%j.out

export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export OMP_NUM_THREADS=30

srun python3 main_base.py


