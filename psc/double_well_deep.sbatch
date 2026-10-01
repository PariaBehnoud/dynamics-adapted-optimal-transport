#!/bin/bash
#SBATCH --job-name=dw_deep
#SBATCH -A YOUR_ALLOCATION
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=deep_%A_%a.out
#SBATCH --error=deep_%A_%a.err

set -euo pipefail
module purge
module load anaconda3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p dw_out
python3 double_well_PRE.py --cell "${SLURM_ARRAY_TASK_ID}" --outdir dw_out
