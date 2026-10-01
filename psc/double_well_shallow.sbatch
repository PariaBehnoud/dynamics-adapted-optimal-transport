#!/bin/bash
#SBATCH --job-name=dw_shallow
#SBATCH -A YOUR_ALLOCATION
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=shallow_%A_%a.out
#SBATCH --error=shallow_%A_%a.err

set -euo pipefail
module purge
module load anaconda3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p dw_shallow

F0S=(0.308 0.308 0.308 0.308 0.462 0.462 0.462 0.462)
TAUS=(0.200 0.100 0.050 0.025 0.200 0.100 0.050 0.025)

python3 double_well_PRE.py --a 0.25 \
  --F0 "${F0S[$SLURM_ARRAY_TASK_ID]}" \
  --tau "${TAUS[$SLURM_ARRAY_TASK_ID]}" \
  --outdir dw_shallow
