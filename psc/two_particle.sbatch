#!/bin/bash
#SBATCH --job-name=two_bead
#SBATCH -A YOUR_ALLOCATION
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=two_bead_%j.out
#SBATCH --error=two_bead_%j.err

set -euo pipefail
module purge
module load anaconda3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 two_particle_PRE.py --tau-study
