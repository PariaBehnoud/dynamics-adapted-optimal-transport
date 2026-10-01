r"""
================================================================================
TWO-BEAD (2D) CONFIGURATION-SPACE VALIDATION OF THE DIFFUSION-CONSISTENT
REGULARIZED JKO SCHEME.   Produces Fig. 3 of the manuscript.
================================================================================

Model (Sec. "Two-bead configuration-space evolution"):
Bead 1 is anchored to the wall, bead 2 is joined to bead 1 by a spring and is
driven by the prescribed protocol f(t).  State x = (x1, x2) in R^2,

    U(x1,x2) = (k/2) x1^2 + (k/2)(x2 - x1)^2 = (1/2) x^T K x ,
    K = k [[2, -1], [-1, 1]] ,      F(t) = (0, f(t))^T ,   f(t) = t ,
    dX = (-K X + F(t)) dt + sqrt(2D) dW .

Exact benchmark (Ornstein-Uhlenbeck):
    dm/dt     = -K m + F(t)
    dSigma/dt = -K Sigma - Sigma K + 2 D I
Both are integrated in closed form (and checked against solve_ivp).

Dynamics-adapted endpoint cost over one step [t_k, t_k + tau]:
    c_k(x,y) = (1/2) (y - Phi_tau(x))^T Sigma_tau^{-1} (y - Phi_tau(x))
    Phi_tau(x) = eta_p(t_{k+1}) + e^{-K tau} (x - eta_p(t_k))
    Sigma_tau  = D K^{-1} (I - e^{-2 K tau})

Regularized JKO step, with the diffusion-consistent coefficient
gamma = (1 - eps)/2  so that  D_eff = D (eps + 2 gamma) = D.

--------------------------------------------------------------------------------
WHY THIS RUNS FAST (the separable solver)
--------------------------------------------------------------------------------
K, e^{-K tau} and Sigma_tau share eigenvectors.  Writing u = Q^T x in the
K-eigenbasis (K = Q diag(kappa) Q^T, Q orthogonal, so the Jacobian is 1 and
densities carry over unchanged),

    c_k(x,y) = sum_r ( v_r - a_r u_r - dt_r )^2 / (2 s_r),
    a_r = e^{-kappa_r tau},  s_r = (D/kappa_r)(1 - e^{-2 kappa_r tau}),
    dt  = Q^T [ eta_p(t_{k+1}) - e^{-K tau} eta_p(t_k) ].

The cost is a SUM over modes, so the Gibbs kernel exp(-c/eps) is a PRODUCT:

    Kgibbs[(i1,i2),(j1,j2)] = G1[i1,j1] * G2[i2,j2] .

On a tensor grid in mode coordinates the kernel therefore never has to be
formed.  With m points per mode the dense kernel would need m^4 entries and
m^4 flops per Sinkhorn matvec; the factorized form needs 2 m^2 entries and
2 m^3 flops.  At m = 201 that is 1.6e9 -> 1.6e7 flops per matvec, and 13 GB
-> 0.6 MB of memory.  This is the "separable modal structure" the manuscript
cites in the scaling discussion; --solver dense reproduces the same numbers
the slow way and is used to verify it.

--------------------------------------------------------------------------------
USAGE (PSC)
--------------------------------------------------------------------------------
    python3 two_particle_PRE.py                       # defaults, ~1 min
    python3 two_particle_PRE.py --tau 0.025 --eps 0.05
    python3 two_particle_PRE.py --validate            # separable vs dense
    python3 two_particle_PRE.py --log-domain          # log-sum-exp iteration
    python3 two_particle_PRE.py --tau-study           # tau refinement table

The inner loop is two dense matmuls per Sinkhorn iteration, which BLAS
parallelizes well, so on PSC give it the cores you asked for:

    #!/bin/bash
    #SBATCH -p RM-shared
    #SBATCH -N 1 --ntasks-per-node=8
    #SBATCH -t 02:00:00
    module load anaconda3
    export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
    python3 two_particle_PRE.py --tau-study

Memory is a few hundred MB (the kernel is never formed), so RM-shared is
enough; only --solver dense needs a large-memory node.

RESOLUTION.  Accuracy is set by the grid spacing relative to the ONE-STEP
KERNEL width sqrt(eps * s_r), not relative to the width of the density.  At
~1.7 points per kernel standard deviation the mean picks up an O(1e-3) error;
at ~1.0 it is O(1e-7) and the variance is grid converged.  --ppk controls
this and defaults to 1.2.  Shrinking tau or eps narrows the kernel and so
demands a finer grid: the printed banner reports the achieved value.

Outputs:
    fig3_twobead.pdf / .png        snapshots + means + variances + covariance
    fig3_twobead_moments.csv       t, means, variances, covariance (JKO + exact)
    fig3_twobead_errors.csv        summary error table
    fig3_twobead_tau_study.csv     (only with --tau-study)
================================================================================
"""

import argparse
import time
import platform
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# COMMAND LINE
# ==============================================================================
ap = argparse.ArgumentParser(description="Two-bead dynamics-adapted JKO validation")
ap.add_argument("--k", type=float, default=1.0, help="spring constant")
ap.add_argument("--D", type=float, default=0.25, help="diffusion coefficient")
ap.add_argument("--T", type=float, default=4.0, help="final time")
ap.add_argument("--tau", type=float, default=0.05, help="JKO time step")
ap.add_argument("--eps", type=float, default=0.05, help="entropic regularization")
ap.add_argument("--gamma", type=float, default=None,
                help="target-entropy coefficient (default: the consistent (1-eps)/2)")
ap.add_argument("--m1", type=int, default=None, help="grid points, soft mode")
ap.add_argument("--m2", type=int, default=None, help="grid points, stiff mode")
ap.add_argument("--ppk", type=float, default=1.2,
                help="grid points per one-step kernel standard deviation "
                     "sqrt(eps*s_r); this, not the density width, is what sets "
                     "accuracy (sets m1, m2 if unset)")
ap.add_argument("--nsigma", type=float, default=5.0, help="domain half-width in sigma")
ap.add_argument("--m0", type=float, nargs=2, default=[0.0, 0.0], help="initial mean")
ap.add_argument("--var0", type=float, default=0.05, help="initial variance (isotropic)")
ap.add_argument("--solver", choices=["separable", "dense"], default="separable")
ap.add_argument("--log-domain", action="store_true",
                help="log-sum-exp stabilized iteration (needed at small eps)")
ap.add_argument("--n-sink", type=int, default=5000, help="max Sinkhorn iterations")
ap.add_argument("--tol", type=float, default=1e-12, help="Sinkhorn tolerance")
ap.add_argument("--validate", action="store_true",
                help="check separable vs dense kernel on the same grid, then exit")
ap.add_argument("--tau-study", action="store_true", help="tau-refinement table")
ap.add_argument("--snapshots", type=float, nargs="*", default=None,
                help="times for the density heat maps")
ap.add_argument("--threads", type=int, default=None, help="BLAS threads")
ap.add_argument("--prefix", type=str, default="fig3_twobead", help="output prefix")
ap.add_argument("--no-figure", action="store_true")
args = ap.parse_args()

if args.threads is not None:
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(args.threads)

k, D, T, tau, eps = args.k, args.D, args.T, args.tau, args.eps


def consistent_gamma(e):
    """Diffusion-consistent entropy coefficient: eps + 2 gamma = 1."""
    if not 0.0 <= e <= 1.0:
        raise ValueError("require 0 <= eps <= 1")
    return 0.5 * (1.0 - e)


gamma = consistent_gamma(eps) if args.gamma is None else args.gamma

# ==============================================================================
# STIFFNESS MATRIX AND ITS EIGENBASIS
# ==============================================================================
K = k * np.array([[2.0, -1.0], [-1.0, 1.0]])
kap, Q = np.linalg.eigh(K)            # kappa_1 < kappa_2 ; Q orthogonal
Kinv = np.linalg.inv(K)
cF = np.array([0.0, 1.0])             # F(t) = cF * f(t), f(t) = t


def expmK(s):
    """e^{-K s} via the eigendecomposition."""
    return Q @ np.diag(np.exp(-kap * s)) @ Q.T


# ------------------------------------------------------------------
# eta_p(t):  d eta/dt = -K eta + cF t,  eta(0) = 0
#   eta(t) = K^{-1} cF t - K^{-2} (I - e^{-K t}) cF
# ------------------------------------------------------------------
K2inv = Kinv @ Kinv


def eta_p(t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    lin = np.outer(t, Kinv @ cF)
    dec = np.array([(np.eye(2) - expmK(ti)) @ K2inv @ cF for ti in t])
    return lin - dec


# ------------------------------------------------------------------
# Exact OU moments.  In the eigenbasis the covariance decouples entrywise:
#   d/dt S_ij = -(kap_i + kap_j) S_ij + 2 D delta_ij
# ------------------------------------------------------------------
m0 = np.asarray(args.m0, dtype=float)
Sigma0 = args.var0 * np.eye(2)
S0u = Q.T @ Sigma0 @ Q
KS = kap[:, None] + kap[None, :]
src = 2.0 * D * np.eye(2)


def exact_moments(t):
    """Exact OU mean m(t) (n,2) and covariance Sigma(t) (n,2,2)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    mean = eta_p(t) + np.array([expmK(ti) @ m0 for ti in t])
    cov = np.empty((t.size, 2, 2))
    for n, ti in enumerate(t):
        E = np.exp(-KS * ti)
        Su = S0u * E + (src / KS) * (1.0 - E)
        cov[n] = Q @ Su @ Q.T
    return mean, cov


# ==============================================================================
# MODE GRID  (auto-sized to contain the trajectory to nsigma)
# ==============================================================================
t_full = np.arange(int(round(T / tau)) + 1) * tau
mean_ex, cov_ex = exact_moments(t_full)
eta_full = eta_p(t_full)
mu_u = mean_ex @ Q                                   # mode-coordinate means
sig_u = np.sqrt(np.array([np.diag(Q.T @ C @ Q) for C in cov_ex]))
lo = (mu_u - args.nsigma * sig_u).min(axis=0)
hi = (mu_u + args.nsigma * sig_u).max(axis=0)

# Grid resolution is set by the ONE-STEP KERNEL width, not by the width of the
# density.  The Gibbs kernel exp(-c/eps) has per-mode conditional standard
# deviation sqrt(eps * s_r) with s_r = (D/kappa_r)(1 - e^{-2 kappa_r tau}).
# If the spacing exceeds that, the kernel is unresolved and the scheme loses
# the exactness of its mean update: at ~1.7 points per kernel std the mean
# error is O(1e-3), at ~1.0 it is O(1e-7) and the variance is grid converged.
TAU_LIST = (0.2, 0.1, 0.05, 0.025)      # used by --tau-study
# A shorter step narrows the kernel (s_r ~ 2 D tau for small tau), so the grid
# must be sized for the SMALLEST tau that will be run, not just for --tau.
tau_grid = min(tau, min(TAU_LIST)) if args.tau_study else tau
s_mode = (D / kap) * (1.0 - np.exp(-2.0 * kap * tau_grid))
w_kernel = np.sqrt(eps * s_mode)
h_target = w_kernel / args.ppk

if args.m1 is None:
    args.m1 = int(np.ceil((hi[0] - lo[0]) / h_target[0])) + 1 | 1
if args.m2 is None:
    args.m2 = int(np.ceil((hi[1] - lo[1]) / h_target[1])) + 1 | 1
m1, m2 = args.m1, args.m2

u1 = np.linspace(lo[0], hi[0], m1)
u2 = np.linspace(lo[1], hi[1], m2)
du1, du2 = u1[1] - u1[0], u2[1] - u2[0]
dA = du1 * du2
U1, U2 = np.meshgrid(u1, u2, indexing="ij")

# initial density on the mode grid (Gaussian, mean Q^T m0, covariance Q^T S0 Q)
mu0u = Q.T @ m0
S0u_inv = np.linalg.inv(S0u)
_d1, _d2 = U1 - mu0u[0], U2 - mu0u[1]
quad = (S0u_inv[0, 0] * _d1 ** 2 + 2 * S0u_inv[0, 1] * _d1 * _d2
        + S0u_inv[1, 1] * _d2 ** 2)
P0 = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(S0u)))


# ==============================================================================
# ONE-STEP KERNEL FACTORS
# ==============================================================================
def kernel_factors(j, tau, eps):
    """Per-mode cost factors C1 (m1 x m1) and C2 (m2 x m2) for step j -> j+1.
    The full cost is C[(i1,i2),(j1,j2)] = C1[i1,j1] + C2[i2,j2]."""
    a = np.exp(-kap * tau)
    s = (D / kap) * (1.0 - np.exp(-2.0 * kap * tau))
    ep0, ep1 = eta_p(j * tau)[0], eta_p((j + 1) * tau)[0]
    dsh = Q.T @ (ep1 - expmK(tau) @ ep0)             # shift in mode coordinates
    C1 = (u1[None, :] - a[0] * u1[:, None] - dsh[0]) ** 2 / (2.0 * s[0])
    C2 = (u2[None, :] - a[1] * u2[:, None] - dsh[1]) ** 2 / (2.0 * s[1])
    # row shifts are row-dependent additive constants: they do not change the OT
    C1 = C1 - C1.min(axis=1, keepdims=True)
    C2 = C2 - C2.min(axis=1, keepdims=True)
    return C1, C2


# ==============================================================================
# GENERALIZED SINKHORN
# ==============================================================================
def sinkhorn_separable(p, C1, C2, eps, gamma, n_sink, tol):
    """Standard (multiplicative) generalized Sinkhorn with a factorized kernel.
    Kb = G1 @ b @ G2.T ,  K^T a = G1.T @ a @ G2 ."""
    lam = gamma / eps
    G1, G2 = np.exp(-C1 / eps), np.exp(-C2 / eps)
    b = np.ones_like(p)
    for it in range(n_sink):
        a = p / np.maximum(G1 @ b @ G2.T, 1e-300)
        z = np.maximum(G1.T @ a @ G2, 1e-300)
        q = z ** (1.0 / (1.0 + lam))
        q /= q.sum()
        bn = q / z
        done = np.max(np.abs(bn - b)) < tol * max(1.0, np.max(np.abs(b)))
        b = bn
        if done:
            break
    return q, it + 1


def _logmm(M, X, transpose=False):
    """log-domain matmul: out[i,c] = logsumexp_j ( M[i,j] + X[j,c] )."""
    A = M.T if transpose else M
    return logsumexp(A[:, :, None] + X[None, :, :], axis=1)


def sinkhorn_separable_log(p, C1, C2, eps, gamma, n_sink, tol):
    """Log-sum-exp stabilized version of the same iteration."""
    lam = gamma / eps
    M1, M2 = -C1 / eps, -C2 / eps
    logp = np.log(np.maximum(p, 1e-300))
    logb = np.zeros_like(p)
    for it in range(n_sink):
        # log(K b): contract mode 2 then mode 1
        tmp = _logmm(M2, logb.T).T                    # over j2 -> i2
        logKb = _logmm(M1, tmp)                       # over j1 -> i1
        loga = logp - logKb
        tmp = _logmm(M2, loga.T, transpose=True).T    # over i2 -> j2
        logz = _logmm(M1, tmp, transpose=True)        # over i1 -> j1
        logq = logz / (1.0 + lam)
        logq -= logsumexp(logq)
        lbn = logq - logz
        done = np.max(np.abs(lbn - logb)) < tol / eps
        logb = lbn
        if done:
            break
    return np.exp(logq), it + 1


def sinkhorn_dense(p, C1, C2, eps, gamma, n_sink, tol):
    """Reference: build the full (m1 m2) x (m1 m2) kernel explicitly."""
    lam = gamma / eps
    M = m1 * m2
    C = (C1[:, None, :, None] + C2[None, :, None, :]).reshape(M, M)
    Kg = np.exp(-C / eps)
    pf = p.reshape(M)
    b = np.ones(M)
    for it in range(n_sink):
        a = pf / np.maximum(Kg @ b, 1e-300)
        z = np.maximum(Kg.T @ a, 1e-300)
        q = z ** (1.0 / (1.0 + lam))
        q /= q.sum()
        bn = q / z
        done = np.max(np.abs(bn - b)) < tol * max(1.0, np.max(np.abs(b)))
        b = bn
        if done:
            break
    return q.reshape(m1, m2), it + 1


SOLVERS = {"separable": sinkhorn_separable, "dense": sinkhorn_dense}


def run_jko(tau, eps, gamma, n_steps, solver="separable", log_domain=False,
            store_all=True, n_sink=5000, tol=1e-12):
    """Evolve the mode-grid density by the regularized JKO scheme."""
    if log_domain and solver == "separable":
        step = sinkhorn_separable_log
    else:
        step = SOLVERS[solver]
    p = P0 * dA
    p = p / p.sum()
    out = [p.copy()] if store_all else None
    iters = []
    for j in range(n_steps):
        C1, C2 = kernel_factors(j, tau, eps)
        p, it = step(p, C1, C2, eps, gamma, n_sink, tol)
        iters.append(it)
        if store_all:
            out.append(p.copy())
    return (np.array(out) if store_all else p), float(np.mean(iters))


# ==============================================================================
# MOMENTS  (mode grid -> physical coordinates)
# ==============================================================================
def moments_from_masses(p):
    """p: (..., m1, m2) probability masses. Returns physical mean (...,2)
    and covariance (...,2,2)."""
    single = (p.ndim == 2)
    P = p[None] if single else p
    w1 = (P.sum(axis=2) * u1).sum(axis=1)
    w2 = (P.sum(axis=1) * u2).sum(axis=1)
    mu = np.stack([w1, w2], axis=1)
    d1 = u1[None, :, None] - mu[:, 0, None, None]
    d2 = u2[None, None, :] - mu[:, 1, None, None]
    c11 = (P * d1 ** 2).sum(axis=(1, 2))
    c22 = (P * d2 ** 2).sum(axis=(1, 2))
    c12 = (P * d1 * d2).sum(axis=(1, 2))
    Su = np.empty((P.shape[0], 2, 2))
    Su[:, 0, 0], Su[:, 1, 1], Su[:, 0, 1], Su[:, 1, 0] = c11, c22, c12, c12
    mx = mu @ Q.T
    Sx = np.einsum("ab,nbc,dc->nad", Q, Su, Q)
    return (mx[0], Sx[0]) if single else (mx, Sx)


# ==============================================================================
# BANNER
# ==============================================================================
print("=" * 79)
print("TWO-BEAD CONFIGURATION-SPACE VALIDATION OF THE REGULARIZED JKO SCHEME")
print("=" * 79)
print(f"k = {k}, D = {D}, K = k[[2,-1],[-1,1]], f(t) = t applied to bead 2")
print(f"kappa = {kap.round(6)}   (soft, stiff);  equilibrium Sigma = D K^-1 = "
      f"{(D * Kinv).round(4).tolist()}")
print(f"m(0) = {m0.tolist()}, Sigma(0) = {args.var0} I, T = {T}, tau = {tau}")
print(f"eps = {eps}, gamma = {gamma:g}, eps + 2 gamma = {eps + 2 * gamma:.3f}")
print(f"mode grid: {m1} x {m2} = {m1 * m2} points, "
      f"u1 in [{lo[0]:.2f}, {hi[0]:.2f}], u2 in [{lo[1]:.2f}, {hi[1]:.2f}]")
print(f"kernel std sqrt(eps*s) = {w_kernel.round(5).tolist()};  spacing "
      f"{du1:.5f}, {du2:.5f}  ->  {w_kernel[0]/du1:.2f}, {w_kernel[1]/du2:.2f} "
      f"points per kernel std  (want >~ 1)")
print(f"solver = {args.solver}{' (log-domain)' if args.log_domain else ''}, "
      f"Sinkhorn tol = {args.tol:g}, max {args.n_sink} it")
print(f"platform: {platform.platform()} | numpy {np.__version__}")

# ==============================================================================
# CONSISTENCY CHECKS ON THE CLOSED FORMS
# ==============================================================================
tc = np.linspace(0, T, 41)
sol = solve_ivp(lambda t, y: (-K @ y[:2] + cF * t).tolist()
                + (-K @ y[2:].reshape(2, 2) - y[2:].reshape(2, 2) @ K
                   + 2 * D * np.eye(2)).ravel().tolist(),
                (0.0, T), np.concatenate([m0, Sigma0.ravel()]), t_eval=tc,
                rtol=1e-12, atol=1e-14, method="DOP853").y
m_num, S_num = sol[:2].T, sol[2:].T.reshape(-1, 2, 2)
m_cf, S_cf = exact_moments(tc)
print("\n" + "-" * 79)
print("[0] CLOSED-FORM CHECKS (analytic vs solve_ivp, DOP853 rtol 1e-12)")
print("-" * 79)
print(f"max |m_closed  - m_numeric|     = {np.max(np.abs(m_cf - m_num)):.2e}")
print(f"max |Sigma_closed - Sigma_num|  = {np.max(np.abs(S_cf - S_num)):.2e}")

# ==============================================================================
# --validate : separable kernel vs full dense kernel, same grid
# ==============================================================================
if args.validate:
    print("\n" + "-" * 79)
    print("[V] SEPARABLE vs DENSE KERNEL (identical grid, identical cost)")
    print("-" * 79)
    if m1 * m2 > 4000:
        print(f"grid is {m1}x{m2} = {m1*m2}; the dense kernel would need "
              f"{(m1*m2)**2*8/1e9:.1f} GB. Re-run --validate with smaller "
              f"--m1/--m2 (e.g. --m1 45 --m2 25).")
        raise SystemExit(1)
    ns = min(10, int(round(T / tau)))
    t0 = time.perf_counter()
    Pa, ia = run_jko(tau, eps, gamma, ns, solver="separable")
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    Pb, ib = run_jko(tau, eps, gamma, ns, solver="dense")
    tb = time.perf_counter() - t0
    t0 = time.perf_counter()
    Pc, ic = run_jko(tau, eps, gamma, ns, solver="separable", log_domain=True)
    tc_ = time.perf_counter() - t0
    print(f"{ns} steps on a {m1} x {m2} grid ({m1*m2} points)")
    print(f"  separable   : {ta:6.2f} s, {ia:.0f} it/step")
    print(f"  dense       : {tb:6.2f} s, {ib:.0f} it/step "
          f"({(m1*m2)**2*8/1e6:.0f} MB kernel)")
    print(f"  log-domain  : {tc_:6.2f} s, {ic:.0f} it/step")
    print(f"  max |P_separable - P_dense|      = {np.max(np.abs(Pa - Pb)):.3e}")
    print(f"  max |P_separable - P_log-domain| = {np.max(np.abs(Pa - Pc)):.3e}")
    print(f"  speed-up separable over dense    = {tb / ta:.1f}x")
    raise SystemExit(0)

# ==============================================================================
# MAIN RUN
# ==============================================================================
n_steps = int(round(T / tau))
t0 = time.perf_counter()
Pt, avg_it = run_jko(tau, eps, gamma, n_steps, solver=args.solver,
                     log_domain=args.log_domain, n_sink=args.n_sink, tol=args.tol)
wall = time.perf_counter() - t0
t_grid = np.arange(n_steps + 1) * tau

m_jko, S_jko = moments_from_masses(Pt)
m_ex, S_ex = exact_moments(t_grid)

err = dict(
    m1=np.max(np.abs(m_jko[:, 0] - m_ex[:, 0])),
    m2=np.max(np.abs(m_jko[:, 1] - m_ex[:, 1])),
    v1=np.max(np.abs(S_jko[:, 0, 0] - S_ex[:, 0, 0])),
    v2=np.max(np.abs(S_jko[:, 1, 1] - S_ex[:, 1, 1])),
    c12=np.max(np.abs(S_jko[:, 0, 1] - S_ex[:, 0, 1])),
)

print("\n" + "-" * 79)
print(f"[1] ERRORS vs EXACT OU, max over t in [0,{T:g}]  "
      f"(tau={tau}, eps={eps}, gamma={gamma:g}, grid {m1}x{m2})")
print("-" * 79)
print(f"{'quantity':<22}{'max abs error':>16}{'final JKO':>14}{'final exact':>14}")
rows = [("<x1>", err["m1"], m_jko[-1, 0], m_ex[-1, 0]),
        ("<x2>", err["m2"], m_jko[-1, 1], m_ex[-1, 1]),
        ("Var(x1)", err["v1"], S_jko[-1, 0, 0], S_ex[-1, 0, 0]),
        ("Var(x2)", err["v2"], S_jko[-1, 1, 1], S_ex[-1, 1, 1]),
        ("Cov(x1,x2)", err["c12"], S_jko[-1, 0, 1], S_ex[-1, 0, 1])]
for nm, e, a, b in rows:
    print(f"{nm:<22}{e:>16.3e}{a:>14.6f}{b:>14.6f}")
print(f"\nSinkhorn iterations per step: {avg_it:.1f}    "
      f"wall clock: {wall:.1f} s ({1e3*wall/n_steps:.0f} ms/step)")
print(f"equilibrium covariance D K^-1 = "
      f"{(D*Kinv)[0,0]:.4f}, {(D*Kinv)[0,1]:.4f}, {(D*Kinv)[1,1]:.4f} "
      f"(Var x1, Cov, Var x2)")

# ==============================================================================
# OPTIONAL TAU REFINEMENT
# ==============================================================================
tau_rows = []
if args.tau_study:
    print("\n" + "-" * 79)
    print(f"[2] TAU REFINEMENT (eps = {eps}, no bias subtraction)")
    print("-" * 79)
    print(f"{'tau':>9}{'max|dVar x1|':>15}{'max|dVar x2|':>15}"
          f"{'max|dCov|':>13}{'ratio(Var x1)':>15}")
    prev = None
    for tt in TAU_LIST:
        n = int(round(T / tt))
        g = consistent_gamma(eps) if args.gamma is None else args.gamma
        Pq, _ = run_jko(tt, eps, g, n, solver=args.solver,
                        log_domain=args.log_domain, n_sink=args.n_sink, tol=args.tol)
        tq = np.arange(n + 1) * tt
        mq, Sq = moments_from_masses(Pq)
        _, Se = exact_moments(tq)
        e1 = np.max(np.abs(Sq[:, 0, 0] - Se[:, 0, 0]))
        e2 = np.max(np.abs(Sq[:, 1, 1] - Se[:, 1, 1]))
        ec = np.max(np.abs(Sq[:, 0, 1] - Se[:, 0, 1]))
        rs = "      ---" if prev is None else f"{prev / e1:>15.2f}"
        print(f"{tt:>9.4f}{e1:>15.3e}{e2:>15.3e}{ec:>13.3e}{rs}")
        tau_rows.append([tt, e1, e2, ec])
        prev = e1

# ==============================================================================
# CSV OUTPUT
# ==============================================================================
np.savetxt(f"{args.prefix}_moments.csv",
           np.column_stack([t_grid, m_jko, m_ex,
                            S_jko[:, 0, 0], S_jko[:, 1, 1], S_jko[:, 0, 1],
                            S_ex[:, 0, 0], S_ex[:, 1, 1], S_ex[:, 0, 1]]),
           delimiter=",", comments="",
           header="t,m1_jko,m2_jko,m1_exact,m2_exact,"
                  "var1_jko,var2_jko,cov_jko,var1_exact,var2_exact,cov_exact")
np.savetxt(f"{args.prefix}_errors.csv",
           np.array([[tau, eps, gamma, m1, m2, err["m1"], err["m2"],
                      err["v1"], err["v2"], err["c12"], avg_it, wall]]),
           delimiter=",", comments="",
           header="tau,eps,gamma,m1,m2,err_mean1,err_mean2,err_var1,err_var2,"
                  "err_cov,sinkhorn_it_per_step,wall_s")
if tau_rows:
    np.savetxt(f"{args.prefix}_tau_study.csv", np.array(tau_rows), delimiter=",",
               comments="", header=f"tau,err_var1,err_var2,err_cov (eps={eps})")

if args.no_figure:
    print("\n(figure suppressed by --no-figure)")
    raise SystemExit(0)

# ==============================================================================
# FIGURE 3
# ==============================================================================
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42,
})

# perceptually ordered, prints legibly in greyscale
CMAP = LinearSegmentedColormap.from_list(
    "dens", ["#FFFFFF", "#DCE9F2", "#93C4DE", "#3E8DC0", "#1B4F8C", "#0B2545"])
C_JKO = {"x1": "#0072B2", "x2": "#D55E00"}
EXACT = dict(color="k", ls=(0, (4, 2.5)), lw=0.9)

snaps = args.snapshots if args.snapshots else [0.0, 0.25, 1.0, 2.0, T]
snaps = [s for s in snaps if s <= T + 1e-12]
idx = [int(round(s / tau)) for s in snaps]


def density_on_physical_grid(p, npix=260, pad=0.6, window=None):
    """Interpolate the mode-grid density onto a regular (x1,x2) grid.
    window = (centre, half_width) restricts the output to that physical box;
    otherwise the whole rotated mode domain is covered."""
    P = p / dA                                        # masses -> density
    f = RegularGridInterpolator((u1, u2), P, bounds_error=False, fill_value=0.0)
    if window is None:
        corners = np.array([[a, b] for a in (u1[0], u1[-1]) for b in (u2[0], u2[-1])])
        xc = corners @ Q.T
        x1g = np.linspace(xc[:, 0].min() - pad, xc[:, 0].max() + pad, npix)
        x2g = np.linspace(xc[:, 1].min() - pad, xc[:, 1].max() + pad, npix)
    else:
        ctr, hw = window
        x1g = np.linspace(ctr[0] - hw[0], ctr[0] + hw[0], npix)
        x2g = np.linspace(ctr[1] - hw[1], ctr[1] + hw[1], npix)
    X1, X2 = np.meshgrid(x1g, x2g, indexing="ij")
    pts = np.stack([X1.ravel(), X2.ravel()], axis=1) @ Q     # x -> u
    return x1g, x2g, f(pts).reshape(npix, npix)


def ellipse(mean, cov, nsig=1.0, n=240):
    th = np.linspace(0, 2 * np.pi, n)
    w, V = np.linalg.eigh(cov)
    pts = (V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.array([np.cos(th), np.sin(th)])))
    return mean[0] + nsig * pts[0], mean[1] + nsig * pts[1]


ns = len(idx)
fig = plt.figure(figsize=(7.1, 5.6))

# ---- row 1: density heat maps ------------------------------------------------
# Each panel is windowed on the exact mean, with the SAME physical window size,
# so the growth and shearing of the density are directly comparable.  Colour is
# normalized per panel (the peak height falls by more than an order of magnitude
# between the first and last snapshot), so these show shape, not absolute height.
Seq = D * Kinv
hw = 3.0 * np.sqrt(np.diag(Seq))

gs0 = fig.add_gridspec(1, ns, left=0.062, right=0.905, top=0.955, bottom=0.605,
                       wspace=0.14)
for c, i in enumerate(idx):
    ax = fig.add_subplot(gs0[0, c])
    ctr = m_ex[i]
    x1g, x2g, Z = density_on_physical_grid(
        Pt[i], npix=200, window=(ctr, hw))
    im = ax.pcolormesh(x1g, x2g, (Z / max(Z.max(), 1e-300)).T, cmap=CMAP,
                       vmin=0, vmax=1, shading="auto", rasterized=True)
    for nsig, lw in ((1.0, 1.0), (2.0, 0.7)):
        ex, ey = ellipse(m_ex[i], S_ex[i], nsig)
        ax.plot(ex, ey, color="#D55E00", lw=lw, ls=(0, (3.2, 2.0)))
    ax.plot(*m_ex[i], "+", color="#D55E00", ms=5, mew=1.0)
    ax.set_xlim(ctr[0] - hw[0], ctr[0] + hw[0])
    ax.set_ylim(ctr[1] - hw[1], ctr[1] + hw[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(rf"$t={i*tau:g}$", pad=3)
    ax.set_xlabel(r"$x_1$", labelpad=1.5)
    ax.tick_params(labelsize=7, pad=1.5)
    if c == 0:
        ax.set_ylabel(r"$x_2$", labelpad=2)
        ax.text(0.05, 0.955, "(a)", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", color="0.15")

cax = fig.add_axes([0.918, 0.605, 0.014, 0.35])
cb = fig.colorbar(im, cax=cax)
cb.set_label("density (per panel max)", fontsize=7.2, labelpad=3)
cb.ax.tick_params(labelsize=6.8, length=2)
cb.set_ticks([0, 0.5, 1.0])

fig.text(0.49, 0.565,
         r"Regularized JKO density $P(x_1,x_2,t)$ in a co-moving window of fixed size; "
         r"dashed: exact OU $1\sigma$, $2\sigma$ ellipses",
         ha="center", va="center", fontsize=7.2, color="0.25")

# ---- row 2: means, variances, covariance -------------------------------------
gs1 = fig.add_gridspec(1, 3, left=0.062, right=0.985, top=0.475, bottom=0.085,
                       wspace=0.30)
axb, axc, axd = (fig.add_subplot(gs1[0, j]) for j in range(3))

axb.plot(t_grid, m_jko[:, 0], color=C_JKO["x1"], lw=1.7)
axb.plot(t_grid, m_jko[:, 1], color=C_JKO["x2"], lw=1.7)
axb.plot(t_grid, m_ex[:, 0], **EXACT)
axb.plot(t_grid, m_ex[:, 1], **EXACT)
axb.set_xlabel(r"$t$"); axb.set_ylabel(r"mean displacement")
axb.set_xlim(0, T)
axb.legend(handles=[Line2D([], [], color=C_JKO["x1"], lw=1.7, label=r"$\langle x_1\rangle$"),
                    Line2D([], [], color=C_JKO["x2"], lw=1.7, label=r"$\langle x_2\rangle$"),
                    Line2D([], [], label="Exact OU", **EXACT)],
           loc="upper left", bbox_to_anchor=(0.0, 0.84), handlelength=1.9)
axb.text(0.04, 0.965, "(b)", transform=axb.transAxes, fontsize=10,
         fontweight="bold", va="top")

axc.plot(t_grid, S_jko[:, 0, 0], color=C_JKO["x1"], lw=1.7)
axc.plot(t_grid, S_jko[:, 1, 1], color=C_JKO["x2"], lw=1.7)
axc.plot(t_grid, S_ex[:, 0, 0], **EXACT)
axc.plot(t_grid, S_ex[:, 1, 1], **EXACT)
for val, col in ((Seq[0, 0], C_JKO["x1"]), (Seq[1, 1], C_JKO["x2"])):
    axc.axhline(val, color=col, lw=0.6, ls=":", alpha=0.85)
axc.set_xlabel(r"$t$"); axc.set_ylabel("variance")
axc.set_xlim(0, T); axc.set_ylim(0, 0.56)
axc.legend(handles=[Line2D([], [], color=C_JKO["x1"], lw=1.7, label=r"$\mathrm{Var}(x_1)$"),
                    Line2D([], [], color=C_JKO["x2"], lw=1.7, label=r"$\mathrm{Var}(x_2)$"),
                    Line2D([], [], label="Exact OU", **EXACT)],
           loc="lower right", handlelength=1.9)
axc.text(0.04, 0.955, "(c)", transform=axc.transAxes, fontsize=10,
         fontweight="bold", va="top")

axd.plot(t_grid, S_jko[:, 0, 1], color="#009E73", lw=1.7)
axd.plot(t_grid, S_ex[:, 0, 1], **EXACT)
axd.axhline(Seq[0, 1], color="#009E73", lw=0.6, ls=":", alpha=0.85)
axd.set_xlabel(r"$t$"); axd.set_ylabel(r"$\mathrm{Cov}(x_1,x_2)$")
axd.set_xlim(0, T); axd.set_ylim(0, Seq[0, 1] * 1.22)
axd.text(T * 0.5, Seq[0, 1] * 1.035,
         r"equilibrium $D\mathbf{K}^{-1}_{12}$", color="#009E73",
         fontsize=7.2, ha="center", va="bottom")
axd.legend(handles=[Line2D([], [], color="#009E73", lw=1.7, label="Regularized JKO"),
                    Line2D([], [], label="Exact OU", **EXACT)],
           loc="lower right", handlelength=1.9)
axd.text(0.04, 0.955, "(d)", transform=axd.transAxes, fontsize=10,
         fontweight="bold", va="top")
fig.savefig(f"{args.prefix}.pdf", bbox_inches="tight")
fig.savefig(f"{args.prefix}.png", dpi=600, bbox_inches="tight")

print("\n" + "-" * 79)
print("FILES WRITTEN")
print("-" * 79)
for f in (f"{args.prefix}.pdf", f"{args.prefix}.png",
          f"{args.prefix}_moments.csv", f"{args.prefix}_errors.csv"):
    print("   " + f)
if tau_rows:
    print(f"   {args.prefix}_tau_study.csv")
