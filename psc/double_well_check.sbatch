#!/bin/bash
#SBATCH --job-name=dw_check
#SBATCH -A YOUR_ALLOCATION
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=check_%j.out
#SBATCH --error=check_%j.err

set -euo pipefail
module purge
module load anaconda3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 double_well_PRE.py --verify
