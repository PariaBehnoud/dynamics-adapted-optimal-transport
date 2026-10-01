# Dynamics-Adapted Optimal Transport for Driven Overdamped Systems

Code supporting the manuscript **"Dynamics-Adapted Optimal Transport for Driven Overdamped Systems."**

## Contents

- `one_particle.py`  
  One-bead harmonic validation against the exact Ornstein–Uhlenbeck solution and Langevin simulation.  
  Produces the one-bead validation figure and CSV summaries.

- `two_particle.py`  
  Two-bead configuration-space validation using the separable modal kernel.  
  Produces `fig3_twobead.pdf/.png` and quantitative mean/covariance diagnostics.

- `double_well.py`  
  Driven nonlinear double-well benchmark. Compares STA, STA+J, standard non-autonomous JKO,
  local-Gaussian, flow-map-only, direct STA, and direct STA+J against a direct Fokker–Planck reference.
  Supports deep- and shallow-well studies and generates the double-well tables/figures.

- `endpoint_cost.py`  
  Compares nonlinear short-time endpoint-cost approximations with a numerical two-point
  boundary-value-problem reference. The BVP calculation is used as an independent numerical
  reference; for the nonconvex problem, a single BVP solve is not claimed to prove the global minimum.

- `ou_regularized_consistency_check.py`  
  Small standalone check of the diffusion-consistency relation
  `epsilon + 2 gamma = 1` for the Ornstein–Uhlenbeck problem.

- `psc/`  
  Optional Slurm templates for reproducing the more expensive calculations on a cluster.

## Python environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical runs

### One-bead validation

```bash
python one_particle.py
```

### Two-bead validation

```bash
python two_particle.py
```

For the time-step refinement study:

```bash
python two_particle.py --tau-study
```

### Double-well solver checks

```bash
python double_well.py --verify
```

### One deep-well calculation

```bash
python double_well.py --a 1.0 --F0 1.8 --tau 0.05 --outdir dw_out
```

### One shallow-well calculation

```bash
python double_well.py --a 0.25 --F0 0.462 --tau 0.05 --outdir dw_shallow
```

### Endpoint-cost/BVP comparison

```bash
python endpoint_cost.py
```

## Cluster runs

The scripts in `psc/` provide Slurm templates for the computationally intensive calculations.
Replace `YOUR_ALLOCATION` with the appropriate allocation/account name for your computing environment before submitting the jobs.
