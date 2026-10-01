#!/bin/bash
set -euo pipefail
module purge
module load anaconda3
export MPLBACKEND=Agg

python3 double_well_PRE.py --collect --outdir dw_out
python3 double_well_PRE.py --collect --outdir dw_shallow --a 0.25

echo "Deep well:    dw_out/table2.tex and dw_out/fig_doublewell.pdf"
echo "Shallow well: dw_shallow/table2.tex and dw_shallow/fig_doublewell.pdf"
