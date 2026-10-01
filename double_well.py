r"""
================================================================================
DRIVEN NON-QUADRATIC DOUBLE-WELL BENCHMARK
Produces Table 2 and the double-well figure of the manuscript.
================================================================================

    U(x) = (x^2 - 1)^2 ,   D = 0.25 ,   V_t(x) = U(x) - F(t) x
    F(t) = F0 t/tau_r for t <= tau_r, else F0 ;  tau_r = 2, F0 in {1.2, 1.8}
    P(x,0) = N(-1, 0.16^2) ,  x in [-3,3] ,  T = 6

The static barrier of the tilted well disappears at F_c = 8/(3 sqrt 3) ~ 1.5396,
so F0 = 1.2 is subcritical (a barrier remains) and F0 = 1.8 is supercritical
(the barrier vanishes during the ramp).

--------------------------------------------------------------------------------
THE SEVEN METHODS COMPARED  (all against a direct Fokker-Planck reference)
--------------------------------------------------------------------------------
Minimizing-movement (generalized Sinkhorn) schemes:
  STA        c^STA, Eq. (STA_kernel): the primary dynamics-adapted kernel
  STA+J      c^STA,J: STA minus (tau/2) * line-averaged Laplacian of V
  JKO        standard non-autonomous JKO with the tilted potential V_{t_k}
  LG         local-Gaussian kernel, drift linearized about the source point
  FM         flow-map-only kernel: free-diffusion Gaussian centred on Phi_tau
Direct propagation (row-normalized kernel, no transport problem):
  dirSTA     K ~ exp(-c^STA)
  dirSTA+J   K ~ exp(-c^STA,J)

The two direct baselines separate the effect of the Jacobian term from the
effect of the minimizing-movement update itself.

--------------------------------------------------------------------------------
IMPLEMENTATION NOTES THAT MATTER
--------------------------------------------------------------------------------
1. LOG-DOMAIN SINKHORN IS MANDATORY HERE.  At eps = 0.02 the Gibbs kernel
   exp(-C/eps) underflows on 60-85% of its entries and several hundred of its
   columns are identically zero in double precision, so the multiplicative
   iteration would be running on the underflow clamp rather than on the kernel.
   Every Sinkhorn method below therefore runs in log-sum-exp form.

2. THE STANDARD-JKO COMPARISON IS RUN IN THE SAME SOLVER.  Writing the
   non-autonomous JKO functional (multiplied by the positive constant 2D) as

       (1/(4 D tau)) W_2^2 + (1/(2D)) \int V_{t_k} P + (1/2) \int P log P

   it is the same regularized problem with cost |x-y|^2/(4 D tau) and an extra
   term linear in the target.  Carrying that term through the optimality
   conditions gives

       q  ~  z^{1/(1+lambda)} * exp( -W / (eps (1+lambda)) ),  lambda = gamma/eps,

   with W_j the coefficient of q_j.  Setting W = 0 recovers the iteration of
   Eq. (gen_sinkhorn).  Both schemes therefore get the same entropic treatment
   and the same diffusion-consistent gamma = (1-eps)/2; the comparison is not
   biased by the regularization.

3. GRID RESOLUTION IS SET BY THE KERNEL WIDTH.  The Gibbs kernel has width
   ~ sqrt(2 D eps tau); if the spacing exceeds it the scheme degrades in a way
   that looks like a modelling error but is not.  The banner reports the ratio
   for every run.

4. THE FOKKER-PLANCK REFERENCE is a Scharfetter-Gummel exponential-fitting
   finite-volume discretization with Crank-Nicolson stepping and no-flux
   boundaries.  It is second-order in dx (verified: error ratios 3.99, 3.98,
   3.93 under successive halving) and conserves mass to 1e-14.  It is solved on
   a grid 4x finer than the transport grid, whose nodes are an exact subset, so
   restricting it to the transport grid introduces no interpolation error.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python3 double_well_PRE.py --verify           # solver checks only
    python3 double_well_PRE.py --F0 1.2 --tau 0.05   # one cell of the table
    python3 double_well_PRE.py --sweep            # the whole table + figure

On PSC the sweep parallelizes cleanly as an array job over the eight
(F0, tau) cells:

    #!/bin/bash
    #SBATCH -p RM-shared -N 1 --ntasks-per-node=8 -t 06:00:00
    #SBATCH --array=0-7
    export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
    python3 double_well_PRE.py --cell $SLURM_ARRAY_TASK_ID
    # then:  python3 double_well_PRE.py --collect
================================================================================
"""

import argparse
import json
import os
import time
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import logsumexp

# ==============================================================================
# COMMAND LINE
# ==============================================================================
ap = argparse.ArgumentParser(description="Driven double-well benchmark")
ap.add_argument("--D", type=float, default=0.25)
ap.add_argument("--a", type=float, default=1.0,
                help="well depth: U(x) = a (x^2-1)^2.  a scales the curvature, "
                     "so it sets the validity parameter tau*|V''| of the "
                     "short-time straight-segment approximation.  a=1 is the "
                     "manuscript's well; a<1 is shallower and moves the "
                     "problem into the regime where c^STA is derived.")
ap.add_argument("--T", type=float, default=6.0)
ap.add_argument("--tau_r", type=float, default=2.0, help="ramp rise time")
ap.add_argument("--F0", type=float, default=None)
ap.add_argument("--tau", type=float, default=None)
ap.add_argument("--eps", type=float, default=0.02)
ap.add_argument("--gamma", type=float, default=None,
                help="default: the diffusion-consistent (1-eps)/2")
ap.add_argument("--M", type=int, default=801, help="transport grid points")
ap.add_argument("--refine", type=int, default=4,
                help="FPE grid refinement factor over the transport grid")
ap.add_argument("--dt-fp", type=float, default=1e-3, help="FPE time step")
ap.add_argument("--xlim", type=float, nargs=2, default=[-3.0, 3.0])
ap.add_argument("--nq", type=int, default=8, help="Gauss-Legendre nodes")
ap.add_argument("--n-sink", type=int, default=20000)
ap.add_argument("--tol", type=float, default=1e-11)
ap.add_argument("--methods", type=str, default=None,
                help="comma-separated subset, e.g. STA,JKO")
ap.add_argument("--verify", action="store_true", help="solver checks, then exit")
ap.add_argument("--sweep", action="store_true", help="the full 2 x 4 table")
ap.add_argument("--cell", type=int, default=None, help="one sweep cell, 0..7")
ap.add_argument("--collect", action="store_true", help="assemble saved cells")
ap.add_argument("--outdir", type=str, default="dw_out")
ap.add_argument("--no-figure", action="store_true")
args = ap.parse_args()

D, T, tau_r, eps = args.D, args.T, args.tau_r, args.eps
F0_LIST = (1.2, 1.8)
TAU_LIST = (0.2, 0.1, 0.05, 0.025)
METHOD_ORDER = ["STA", "STA+J", "JKO", "LG", "FM", "dirSTA", "dirSTA+J"]
F_C = args.a * 8.0 / (3.0 * np.sqrt(3.0))


def consistent_gamma(e):
    if not 0.0 <= e <= 1.0:
        raise ValueError("require 0 <= eps <= 1")
    return 0.5 * (1.0 - e)


gamma = consistent_gamma(eps) if args.gamma is None else args.gamma

# ==============================================================================
# POTENTIAL, FORCE, GRIDS
# ==============================================================================
AW = args.a
U = lambda z: AW * (z ** 2 - 1.0) ** 2
dU = lambda z: AW * 4.0 * z * (z ** 2 - 1.0)
d2U = lambda z: AW * (12.0 * z ** 2 - 4.0)


def make_force(F0):
    def F(t):
        return F0 * min(t / tau_r, 1.0)
    return F


x = np.linspace(args.xlim[0], args.xlim[1], args.M)
dx = x[1] - x[0]
xf = np.linspace(args.xlim[0], args.xlim[1], (args.M - 1) * args.refine + 1)
dxf = xf[1] - xf[0]
SUB = slice(None, None, args.refine)          # xf[SUB] == x exactly

MU0, SIG0 = -1.0, 0.16
p0_dens = np.exp(-(x - MU0) ** 2 / (2 * SIG0 ** 2)) / np.sqrt(2 * np.pi * SIG0 ** 2)
p0_dens_f = np.exp(-(xf - MU0) ** 2 / (2 * SIG0 ** 2)) / np.sqrt(2 * np.pi * SIG0 ** 2)

GLs, GLw = np.polynomial.legendre.leggauss(args.nq)
GLs = 0.5 * (GLs + 1.0)
GLw = 0.5 * GLw


# ==============================================================================
# FOKKER-PLANCK REFERENCE  (Scharfetter-Gummel + Crank-Nicolson, no flux)
# ==============================================================================
def bernoulli(z):
    out = np.empty_like(z)
    s = np.abs(z) < 1e-8
    out[s] = 1.0 - z[s] / 2.0 + z[s] ** 2 / 12.0
    zl = z[~s]
    out[~s] = zl / np.expm1(zl)
    return out


def _sg_tridiag(xg, Vnode, D):
    h = xg[1] - xg[0]
    zf = (Vnode[1:] - Vnode[:-1]) / D
    Bp, Bm = bernoulli(zf), bernoulli(-zf)
    c = D / h ** 2
    n = xg.size
    lo = np.zeros(n); di = np.zeros(n); up = np.zeros(n)
    di[:-1] += -c * Bp
    up[:-1] += c * Bm
    lo[1:] += c * Bp
    di[1:] += -c * Bm
    return lo, di, up


def fpe_solve(xg, P0, F, T, dt, D, t_out):
    """Return the density at each requested output time (rows)."""
    n = int(round(T / dt))
    outs = np.empty((len(t_out), xg.size))
    want = [int(round(tt / dt)) for tt in t_out]
    P = P0 / (P0.sum() * (xg[1] - xg[0]))
    ptr = 0
    while ptr < len(want) and want[ptr] == 0:
        outs[ptr] = P; ptr += 1
    Uv = U(xg)
    for j in range(n):
        Vn = Uv - F(j * dt) * xg
        Vn1 = Uv - F((j + 1) * dt) * xg
        lo, di, up = _sg_tridiag(xg, 0.5 * (Vn + Vn1), D)
        rhs = P + 0.5 * dt * (di * P
                              + np.concatenate([up[:-1] * P[1:], [0.0]])
                              + np.concatenate([[0.0], lo[1:] * P[:-1]]))
        ab = np.zeros((3, xg.size))
        ab[0, 1:] = -0.5 * dt * up[:-1]
        ab[1, :] = 1.0 - 0.5 * dt * di
        ab[2, :-1] = -0.5 * dt * lo[1:]
        P = solve_banded((1, 1), ab, rhs)
        while ptr < len(want) and want[ptr] == j + 1:
            outs[ptr] = P; ptr += 1
    return outs


# ==============================================================================
# DETERMINISTIC FLOW MAP  (RK4, vectorized over all grid points)
# ==============================================================================
def flow_map(xs, t0, tau, F, nsub=40):
    z = xs.copy()
    h = tau / nsub
    for i in range(nsub):
        t = t0 + i * h
        f1 = -dU(z) + F(t)
        f2 = -dU(z + 0.5 * h * f1) + F(t + 0.5 * h)
        f3 = -dU(z + 0.5 * h * f2) + F(t + 0.5 * h)
        f4 = -dU(z + h * f3) + F(t + h)
        z = z + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return z


# ==============================================================================
# ENDPOINT COSTS.  Each returns (C, W): cost matrix and the linear-in-target
# coefficient vector (W = None when the potential is already inside C).
# ==============================================================================
def _line_integrals(F, tk, want_lap):
    """Gauss-Legendre line integrals of |V'|^2 and, optionally, of Lap V,
    along the straight segment from x_i to x_j."""
    Ft = F(tk)
    Xi, Y = x[:, None], x[None, :]
    q2 = np.zeros((x.size, x.size))
    lap = np.zeros_like(q2) if want_lap else None
    for s, w in zip(GLs, GLw):
        z = Xi + s * (Y - Xi)
        q2 += w * (dU(z) - Ft) ** 2
        if want_lap:
            lap += w * d2U(z)
    return q2, lap


def cost_STA(F, tk, tau, jacobian=False):
    Ft = F(tk)
    V = U(x) - Ft * x
    q2, lap = _line_integrals(F, tk, jacobian)
    C = ((x[None, :] - x[:, None]) ** 2 / (4.0 * D * tau)
         + (V[None, :] - V[:, None]) / (2.0 * D)
         + tau / (4.0 * D) * q2)
    if jacobian:
        C = C - 0.5 * tau * lap
    return C, None


def cost_JKO(F, tk, tau):
    """Standard non-autonomous JKO: quadratic transport plus a tilted-potential
    term that is linear in the target density."""
    C = (x[None, :] - x[:, None]) ** 2 / (4.0 * D * tau)
    W = (U(x) - F(tk) * x) / (2.0 * D)
    return C, W


def cost_LG(F, tk, tau):
    Phi = flow_map(x, tk, tau, F)
    H = d2U(x)
    small = np.abs(H) < 1e-10
    Sig = np.where(small, 2.0 * D * tau, D * (1.0 - np.exp(-2.0 * H * tau)) / np.where(small, 1.0, H))
    C = (x[None, :] - Phi[:, None]) ** 2 / (2.0 * Sig[:, None])
    return C, None


def cost_FM(F, tk, tau):
    Phi = flow_map(x, tk, tau, F)
    C = (x[None, :] - Phi[:, None]) ** 2 / (4.0 * D * tau)
    return C, None


COSTS = {
    "STA":      lambda F, tk, tau: cost_STA(F, tk, tau, jacobian=False),
    "STA+J":    lambda F, tk, tau: cost_STA(F, tk, tau, jacobian=True),
    "JKO":      cost_JKO,
    "LG":       cost_LG,
    "FM":       cost_FM,
    "dirSTA":   lambda F, tk, tau: cost_STA(F, tk, tau, jacobian=False),
    "dirSTA+J": lambda F, tk, tau: cost_STA(F, tk, tau, jacobian=True),
}
DIRECT = {"dirSTA", "dirSTA+J"}


# ==============================================================================
# LOG-DOMAIN GENERALIZED SINKHORN
#   q ~ z^{1/(1+lam)} exp( -W / (eps (1+lam)) ),   lam = gamma/eps
# ==============================================================================
def _lse(A, axis):
    """log-sum-exp along `axis`, in place on A. About 5x faster than
    scipy.special.logsumexp here because it avoids the extra allocations;
    verified to agree with it to 9e-16."""
    m = A.max(axis=axis, keepdims=True)
    np.subtract(A, m, out=A)
    np.exp(A, out=A)
    return np.squeeze(m, axis=axis) + np.log(A.sum(axis=axis))


def sinkhorn_log(logp, C, W, eps, gamma, g0=None, n_sink=20000, tol=1e-11):
    lam = gamma / eps
    Mlog = -(C - C.min(axis=1, keepdims=True)) / eps
    g = np.zeros(C.shape[1]) if g0 is None else g0.copy()
    shift = 0.0 if W is None else W / (eps * (1.0 + lam))
    buf = np.empty_like(Mlog)
    for it in range(n_sink):
        np.add(Mlog, g[None, :] / eps, out=buf)
        f = eps * (logp - _lse(buf, 1))
        np.add(Mlog, f[:, None] / eps, out=buf)
        lz = _lse(buf, 0)
        lq = lz / (1.0 + lam) - shift
        lq -= logsumexp(lq)
        gn = eps * (lq - lz)
        done = np.max(np.abs(gn - g)) < tol * eps
        g = gn
        if done:
            break
    return lq, g, it + 1


def direct_propagate(p, C):
    """Row-normalized kernel propagation, in log domain for safety."""
    Mlog = -(C - C.min(axis=1, keepdims=True))
    Mlog = Mlog - logsumexp(Mlog, axis=1, keepdims=True)      # row-stochastic
    lp = np.log(np.maximum(p, 1e-300))
    return np.exp(logsumexp(Mlog + lp[:, None], axis=0))


# ==============================================================================
# ONE TRANSPORT RUN
# ==============================================================================
def run_method(name, F, tau, t_out, n_sink, tol):
    n_steps = int(round(T / tau))
    want = [int(round(tt / tau)) for tt in t_out]
    outs = np.empty((len(t_out), x.size))
    p = p0_dens * dx
    p /= p.sum()
    ptr = 0
    while ptr < len(want) and want[ptr] == 0:
        outs[ptr] = p / dx; ptr += 1
    g = None
    iters = []
    t0 = time.perf_counter()
    for j in range(n_steps):
        C, W = COSTS[name](F, j * tau, tau)
        if name in DIRECT:
            p = direct_propagate(p, C)
            p /= p.sum()
        else:
            # Cold start each step.  Warm starting from the previous step's
            # potentials was measured to be counter-productive here (iteration
            # counts rose from ~290 to ~590), because the cost matrix changes
            # with the frozen tilted potential V_{t_k} between steps.
            lq, g, it = sinkhorn_log(np.log(np.maximum(p, 1e-300)), C, W,
                                     eps, gamma, g0=None, n_sink=n_sink, tol=tol)
            p = np.exp(lq); p /= p.sum()
            iters.append(it)
        while ptr < len(want) and want[ptr] == j + 1:
            outs[ptr] = p / dx; ptr += 1
    return outs, (float(np.mean(iters)) if iters else 0.0), time.perf_counter() - t0


def kernel_resolution(tau):
    """Points per Gibbs-kernel standard deviation, ~ sqrt(2 D eps tau)."""
    return np.sqrt(2.0 * D * eps * tau) / dx


# ==============================================================================
# VERIFICATION MODE
# ==============================================================================
def verify():
    from scipy.integrate import solve_ivp
    print("=" * 79)
    print("SOLVER VERIFICATION")
    print("=" * 79)

    # (1) FPE against the exact OU solution on a harmonic potential
    global U, dU, d2U
    Usave, dUsave, d2Usave = U, dU, d2U
    k = 1.0
    U = lambda z: 0.5 * k * z ** 2
    m0, v0 = 1.5, 0.05
    Fh = lambda t: 2.0 * np.cos(t)
    xg = np.linspace(-8, 8, 3201)
    P0 = np.exp(-(xg - m0) ** 2 / (2 * v0)) / np.sqrt(2 * np.pi * v0)
    Tv = 4.0
    s = solve_ivp(lambda t, y: [-k * y[0] + Fh(t), -2 * k * y[1] + 2 * D],
                  (0, Tv), [m0, v0], rtol=1e-12, atol=1e-14,
                  method="DOP853", max_step=0.01)
    mE, vE = s.y[0, -1], s.y[1, -1]
    P = fpe_solve(xg, P0, Fh, Tv, 1e-3, D, [Tv])[0]
    h = xg[1] - xg[0]
    mass = P.sum() * h
    m = (xg * P).sum() * h / mass
    v = ((xg - m) ** 2 * P).sum() * h / mass
    Pex = np.exp(-(xg - mE) ** 2 / (2 * vE)) / np.sqrt(2 * np.pi * vE)
    print("\n[1] FPE vs exact Ornstein-Uhlenbeck (harmonic U, cyclic forcing)")
    print(f"    mean err = {abs(m-mE):.2e},  var err = {abs(v-vE):.2e},  "
          f"L1 = {np.abs(P-Pex).sum()*h:.2e},  mass err = {abs(mass-1):.1e}")
    U, dU, d2U = Usave, dUsave, d2Usave

    # (2) FPE self-convergence on the actual double well
    print("\n[2] FPE self-convergence on the double well (F0 = 1.8, t = 6)")
    F = make_force(1.8)
    ref = None
    print(f"    {'M_fp':>8}{'dx':>10}{'L1 vs finest':>15}{'ratio':>9}")
    grids = [(args.M - 1) * r + 1 for r in (1, 2, 4, 8)]
    sols = []
    for Mg in grids:
        xg = np.linspace(args.xlim[0], args.xlim[1], Mg)
        P0 = np.exp(-(xg - MU0) ** 2 / (2 * SIG0 ** 2)) / np.sqrt(2 * np.pi * SIG0 ** 2)
        sols.append((xg, fpe_solve(xg, P0, F, T, args.dt_fp, D, [T])[0]))
    xfin, Pfin = sols[-1]
    prev = None
    for (xg, P), Mg in zip(sols[:-1], grids[:-1]):
        step = (len(xfin) - 1) // (len(xg) - 1)
        e = np.abs(P - Pfin[::step]).sum() * (xg[1] - xg[0])
        rs = "     ---" if prev is None else f"{prev/e:>9.2f}"
        print(f"    {Mg:>8}{xg[1]-xg[0]:>10.5f}{e:>15.3e}{rs}")
        prev = e

    # (3) FPE time-step convergence
    print("\n[3] FPE time-step convergence (M_fp fixed at the production grid)")
    xg = xf
    P0 = p0_dens_f
    base = fpe_solve(xg, P0, F, T, 2.5e-4, D, [T])[0]
    print(f"    {'dt':>10}{'L1 vs dt/4':>14}")
    for dt in (4e-3, 2e-3, 1e-3):
        P = fpe_solve(xg, P0, F, T, dt, D, [T])[0]
        print(f"    {dt:>10.5f}{np.abs(P-base).sum()*dxf:>14.3e}")

    # (4) kernel resolution and underflow at the production parameters
    print(f"\n[4] TRANSPORT GRID  M = {args.M}, dx = {dx:.5f}, eps = {eps}, "
          f"gamma = {gamma:g}")
    print(f"    {'tau':>8}{'kernel std':>13}{'pts/std':>10}"
          f"{'underflow frac':>16}{'zero cols':>11}")
    for tt in TAU_LIST:
        C, _ = cost_STA(make_force(1.8), 2.0, tt)
        Cs = C - C.min(axis=1, keepdims=True)
        frac = float((Cs / eps > 709).mean())
        K = np.exp(-Cs / eps)
        zc = int((K.sum(axis=0) == 0).sum())
        w = np.sqrt(2.0 * D * eps * tt)
        print(f"    {tt:>8.4f}{w:>13.5f}{w/dx:>10.2f}{frac:>16.3f}{zc:>11d}")
    print("    (the zero columns are why every Sinkhorn method here runs in "
          "log domain)")

    # (5) the quadratic limit: on a harmonic potential the LG kernel is exact
    print("\n[5] LG kernel on a harmonic potential reduces to the exact "
          "Ornstein-Uhlenbeck kernel")
    kk = 1.0
    Ue, dUe, d2Ue = U, dU, d2U
    U = lambda z: 0.5 * kk * z ** 2
    dU = lambda z: kk * z
    d2U = lambda z: kk * np.ones_like(z)
    tt = 0.05
    Fz = lambda t: 0.0
    Clg, _ = cost_LG(Fz, 0.0, tt)
    Sig_ex = (D / kk) * (1.0 - np.exp(-2.0 * kk * tt))
    Cex = (x[None, :] - np.exp(-kk * tt) * x[:, None]) ** 2 / (2.0 * Sig_ex)
    print(f"    max |C_LG - C_exact| = {np.max(np.abs(Clg-Cex)):.2e}")
    U, dU, d2U = Ue, dUe, d2Ue

    print(f"\nF_c = 8/(3 sqrt 3) = {F_C:.4f};  F0 = 1.2 subcritical, "
          f"F0 = 1.8 supercritical")


if args.verify:
    verify()
    raise SystemExit(0)


# ==============================================================================
# ONE CELL OF THE TABLE:  a given (F0, tau), all methods
# ==============================================================================
def left_well_population(P):
    """Probability mass at x < 0 (the initially occupied well)."""
    w = np.ones_like(x) * dx
    w[0] = w[-1] = 0.5 * dx
    return float((P * w)[x < 0].sum())


def run_cell(F0, tau, methods, save=True, verbose=True):
    F = make_force(F0)
    n_steps = int(round(T / tau))
    t_out = [j * tau for j in range(n_steps + 1)]

    t0 = time.perf_counter()
    Pfp_f = fpe_solve(xf, p0_dens_f, F, T, args.dt_fp, D, t_out)
    Pfp = Pfp_f[:, SUB]                       # exact restriction, no interpolation
    wall_fp = time.perf_counter() - t0

    wq = np.ones_like(x) * dx
    wq[0] = wq[-1] = 0.5 * dx

    res = {"F0": F0, "tau": tau, "eps": eps, "gamma": gamma, "M": args.M,
           "n_steps": n_steps, "wall_fp": wall_fp, "methods": {}}
    dens = {"t_out": t_out, "x": x.tolist(), "FP": Pfp.tolist()}

    if verbose:
        print(f"\n{'-'*79}\nF0 = {F0}  tau = {tau}  ({n_steps} steps, "
              f"{kernel_resolution(tau):.2f} pts per kernel std)"
              f"   FPE reference: {wall_fp:.1f} s\n{'-'*79}")
        print(f"{'method':<10}{'max E_L1':>12}{'<E_L1>_t':>12}"
              f"{'left well T':>14}{'Sinkhorn it':>13}{'wall [s]':>10}")

    for nm in methods:
        Pm, it, wall = run_method(nm, F, tau, t_out, args.n_sink, args.tol)
        EL1 = (np.abs(Pm - Pfp) * wq[None, :]).sum(axis=1)
        res["methods"][nm] = {
            "max_L1": float(EL1.max()), "mean_L1": float(EL1.mean()),
            "left_well_T": left_well_population(Pm[-1]),
            "sinkhorn_it": it, "wall": wall,
            "EL1_t": EL1.tolist(),
            "left_well_t": [left_well_population(P) for P in Pm]}
        dens[nm] = Pm.tolist()
        if verbose:
            print(f"{nm:<10}{EL1.max():>12.3e}{EL1.mean():>12.3e}"
                  f"{res['methods'][nm]['left_well_T']:>14.4f}"
                  f"{it:>13.0f}{wall:>10.1f}")

    res["left_well_T_FP"] = left_well_population(Pfp[-1])
    res["left_well_t_FP"] = [left_well_population(P) for P in Pfp]
    if verbose:
        print(f"{'FPE ref':<10}{'--':>12}{'--':>12}"
              f"{res['left_well_T_FP']:>14.4f}")

    if save:
        os.makedirs(args.outdir, exist_ok=True)
        tag = f"F{F0:g}_tau{tau:g}".replace(".", "p")
        with open(os.path.join(args.outdir, f"cell_{tag}.json"), "w") as fh:
            json.dump(res, fh)
        np.savez_compressed(os.path.join(args.outdir, f"dens_{tag}.npz"),
                            **{k: np.asarray(v) for k, v in dens.items()})
    return res


METHODS = (args.methods.split(",") if args.methods else METHOD_ORDER)
CELLS = [(F0, tt) for F0 in F0_LIST for tt in TAU_LIST]

print("=" * 79)
print(f"DRIVEN DOUBLE-WELL BENCHMARK   U(x) = {AW:g} (x^2-1)^2")
print("=" * 79)
print(f"D = {D}, T = {T}, ramp rise tau_r = {tau_r}, F_c = {F_C:.4f}")
print(f"P(x,0) = N({MU0}, {SIG0}^2), x in [{args.xlim[0]}, {args.xlim[1]}], "
      f"M = {args.M} (dx = {dx:.5f})")
print(f"eps = {eps}, gamma = (1-eps)/2 = {gamma:g}, eps + 2 gamma = "
      f"{eps + 2*gamma:.2f}")
print(f"FPE reference: M = {xf.size} (dx = {dxf:.6f}), dt = {args.dt_fp}, "
      f"Scharfetter-Gummel + Crank-Nicolson")
print(f"methods: {', '.join(METHODS)}")

if args.cell is not None:
    F0c, tauc = CELLS[args.cell]
    run_cell(F0c, tauc, METHODS)
elif args.sweep:
    for F0c, tauc in CELLS:
        run_cell(F0c, tauc, METHODS)
elif args.F0 is not None or args.tau is not None:
    run_cell(args.F0 if args.F0 is not None else 1.8,
             args.tau if args.tau is not None else 0.05, METHODS)
elif not args.collect:
    print("\nNothing to do: pass --verify, --F0/--tau, --cell N, --sweep "
          "or --collect.")


# ==============================================================================
# COLLECT: assemble the saved cells into Table 2 (LaTeX) and the figure
# ==============================================================================
if args.collect:
    import glob
    files = sorted(glob.glob(os.path.join(args.outdir, "cell_*.json")))
    if not files:
        print(f"\nNo cells found in {args.outdir}/. Run --cell 0..7 (or --sweep) "
              f"first.")
        raise SystemExit(1)
    cells = []
    for fn in files:
        with open(fn) as fh:
            cells.append(json.load(fh))
    cells.sort(key=lambda c: (c["F0"], -c["tau"]))
    present = [m for m in METHOD_ORDER
               if any(m in c["methods"] for c in cells)]

    print("\n" + "=" * 79)
    print(f"TABLE 2   max_t E_L1  /  <E_L1>_t   ({len(cells)} cells from "
          f"{args.outdir}/)")
    print("=" * 79)
    print(f"{'F0':>5}{'tau':>8}" + "".join(f"{m:>18}" for m in present))
    for c in cells:
        row = f"{c['F0']:>5.2f}{c['tau']:>8.4f}"
        for m in present:
            d = c["methods"].get(m)
            row += (f"{d['max_L1']:>8.1e} /{d['mean_L1']:>8.1e}" if d
                    else f"{'--':>18}")
        print(row)

    # best method per cell, which is the question the table has to answer
    print("\nBest method per cell (by max E_L1):")
    for c in cells:
        got = {m: c["methods"][m]["max_L1"] for m in present if m in c["methods"]}
        b = min(got, key=got.get)
        print(f"  F0 = {c['F0']:.2f}, tau = {c['tau']:.4f}:  {b}  "
              f"({got[b]:.3e})" +
              (f"   [c^STA = {got['STA']:.3e}, ratio {got['STA']/got[b]:.2f}]"
               if "STA" in got and b != "STA" else ""))

    # LaTeX
    def _latex_sci(v):
        mant, expo = f"{v:.2e}".split("e")
        return rf"${mant}\times 10^{{{int(expo)}}}$"

    # Keep the deep-well and shallow-well tables distinguishable in LaTeX.
    table_label = ("tab:double_well_errors"
                   if np.isclose(args.a, 1.0)
                   else "tab:double_well_errors_shallow")

    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Maximum and time-averaged $L^1$ density errors for the "
           r"driven double-well calculations, relative to the direct "
           r"Fokker--Planck reference. Each entry reports "
           r"$\max_t E_{L^1}(t)$ / $\langle E_{L^1}\rangle_t$.}",
           rf"\label{{{table_label}}}",
           r"\begin{tabular}{cc|" + "c" * len(present) + "}", r"\hline",
           "$F_0$ & $\\tau$ & " + " & ".join(
               {"STA": r"STA", "STA+J": r"STA+J", "JKO": r"Standard JKO",
                "LG": r"Local Gaussian", "FM": r"Flow-map",
                "dirSTA": r"Direct STA", "dirSTA+J": r"Direct STA+J"}[m]
               for m in present) + r" \\", r"\hline"]

    last = None
    for c in cells:
        if last is not None and c["F0"] != last:
            tex.append(r"\hline")
        last = c["F0"]

        # Preserve the actual matched shallow-well forces (0.308 and 0.462)
        # while still printing deep-well values compactly as 1.2 and 1.8.
        f0_text = f"{c['F0']:.3f}".rstrip("0").rstrip(".")

        tex.append(
            f"{f0_text} & {c['tau']:.3f} & "
            + " & ".join(
                (
                    f"{_latex_sci(c['methods'][m]['max_L1'])} / "
                    f"{_latex_sci(c['methods'][m]['mean_L1'])}"
                    if m in c["methods"] else "--"
                )
                for m in present
            )
            + r" \\"
        )

    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(args.outdir, "table2.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"\nWrote {args.outdir}/table2.tex")

    if args.no_figure:
        raise SystemExit(0)

    # ---------------- figure ----------------
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
        "axes.titlesize": 9, "legend.fontsize": 7.0, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42})
    # Okabe-Ito based; the two methods the discussion turns on (standard JKO and
    # direct STA+J) get the two strongest, most separated hues.
    COL = {"STA": "#0072B2", "STA+J": "#56B4E9", "JKO": "#D55E00",
           "LG": "#009E73", "FM": "#999999", "dirSTA": "#7E57C2",
           "dirSTA+J": "#CC0066"}
    LW = {m: (2.0 if m in ("JKO", "dirSTA+J") else 1.1) for m in METHOD_ORDER}
    LBL = {"STA": r"$c^{\rm STA}$", "STA+J": r"$c^{\rm STA,J}$",
           "JKO": "standard JKO", "LG": "local Gaussian", "FM": "flow-map",
           "dirSTA": r"direct $c^{\rm STA}$", "dirSTA+J": r"direct $c^{\rm STA,J}$"}

    F0s = sorted({c["F0"] for c in cells})
    Fpick = F0s[-1]
    cand = sorted([c for c in cells if c["F0"] == Fpick],
                  key=lambda c: abs(c["tau"] - 0.05))
    cref = cand[0]
    tag = f"F{cref['F0']:g}_tau{cref['tau']:g}".replace(".", "p")
    npz = os.path.join(args.outdir, f"dens_{tag}.npz")
    have_dens = os.path.exists(npz)

    fig = plt.figure(figsize=(7.1, 5.4))

    # ---- row 1: density snapshots, with a residual strip under each ----------
    SHOW = [m for m in ("JKO", "dirSTA+J", "STA") if any(m in c["methods"] for c in cells)]
    if have_dens:
        dd = np.load(npz)
        xg = dd["x"]; tt = dd["t_out"]; PF = dd["FP"]
        snaps = [s for s in (0.0, 2.0, 4.0, 6.0) if s <= tt[-1] + 1e-9]
        gs0 = fig.add_gridspec(2, len(snaps), left=0.075, right=0.985,
                               top=0.965, bottom=0.665, hspace=0.07,
                               wspace=0.26, height_ratios=[2.6, 1.0])
        for ci, ts in enumerate(snaps):
            j = int(np.argmin(np.abs(tt - ts)))
            ax = fig.add_subplot(gs0[0, ci])
            ax.fill_between(xg, PF[j], color="0.85", lw=0)
            for mname in SHOW:
                if mname in dd:
                    ax.plot(xg, dd[mname][j], color=COL[mname], lw=1.1)
            ax.set_xlim(-2.2, 2.2); ax.set_xticklabels([])
            ax.set_title(rf"$t={tt[j]:g}$", fontsize=8, pad=2)
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel(r"$P(x,t)$", labelpad=2)
                ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, fontsize=10,
                        fontweight="bold", va="top")
            axr = fig.add_subplot(gs0[1, ci])
            rmax = 0.0
            for mname in SHOW:
                if mname in dd:
                    r = dd[mname][j] - PF[j]
                    axr.plot(xg, r, color=COL[mname], lw=0.9)
                    rmax = max(rmax, float(np.max(np.abs(r))))
            axr.axhline(0, color="0.6", lw=0.5)
            axr.set_xlim(-2.2, 2.2); axr.tick_params(labelsize=7)
            # floor the range so a snapshot where every method is exact (t=0)
            # does not produce degenerate tick labels
            axr.set_ylim(-1.15 * max(rmax, 1e-3), 1.15 * max(rmax, 1e-3))
            axr.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3),
                                 useOffset=False)
            axr.set_xlabel(r"$x$", labelpad=1)
            if ci == 0:
                axr.set_ylabel(r"$P-P_{\rm FP}$", labelpad=2, fontsize=8)

    present_plot = [m for m in METHOD_ORDER if m in cref["methods"]]

    # ---- row 2 ---------------------------------------------------------------
    gs1 = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.555,
                           bottom=0.145, wspace=0.34)
    axb, axc, axd = (fig.add_subplot(gs1[0, j]) for j in range(3))
    tt2 = np.arange(cref["n_steps"] + 1) * cref["tau"]

    # (b) error in the left-well population (the populations themselves are
    #     visually indistinguishable, so the error is what discriminates)
    nF = np.asarray(cref["left_well_t_FP"])
    for m in present_plot:
        e = np.abs(np.asarray(cref["methods"][m]["left_well_t"]) - nF)
        axb.semilogy(tt2[1:], np.maximum(e[1:], 1e-12), color=COL[m], lw=LW[m])
    axb.set_xlabel(r"$t$")
    axb.set_ylabel(r"$|n_{\rm left}-n_{\rm left}^{\rm FP}|$")
    axb.text(0.04, 0.95, "(b)", transform=axb.transAxes, fontsize=10,
             fontweight="bold", va="top")

    # (c) L1 density error in time
    lo = 1e9
    for m in present_plot:
        e = np.asarray(cref["methods"][m]["EL1_t"])
        axc.semilogy(tt2[1:], e[1:], color=COL[m], lw=LW[m])
        lo = min(lo, e[1:].min())
    axc.set_xlabel(r"$t$"); axc.set_ylabel(r"$E_{L^1}(t)$")
    axc.set_ylim(max(lo * 0.5, 1e-6), None)
    axc.text(0.04, 0.95, "(c)", transform=axc.transAxes, fontsize=10,
             fontweight="bold", va="top")

    # (d) time-step convergence
    taus = sorted({c["tau"] for c in cells if c["F0"] == Fpick})
    for m in METHOD_ORDER:
        pts = sorted((c["tau"], c["methods"][m]["max_L1"]) for c in cells
                     if c["F0"] == Fpick and m in c["methods"])
        if len(pts) > 1:
            axd.loglog(*zip(*pts), "o-", ms=3.4, lw=LW[m], color=COL[m])
    if len(taus) > 1:
        ref = [c["methods"]["JKO"]["max_L1"] for c in cells
               if c["F0"] == Fpick and c["tau"] == taus[0] and "JKO" in c["methods"]]
        if ref:
            sl = np.array(taus)
            axd.loglog(sl, ref[0] * sl / taus[0] * 0.42, color="0.45", lw=0.8)
            axd.text(taus[1], ref[0] * taus[1] / taus[0] * 0.26, r"$\propto\tau$",
                     color="0.35", fontsize=8, ha="center")
    axd.set_xlabel(r"time step $\tau$"); axd.set_ylabel(r"$\max_t E_{L^1}$")
    axd.set_xticks(taus)
    axd.set_xticklabels([f"{t:g}" for t in taus])
    axd.xaxis.set_minor_formatter(NullFormatter())
    axd.xaxis.set_minor_locator(plt.NullLocator())
    axd.set_xlim(taus[0] * 0.78, taus[-1] * 1.28)
    axd.text(0.04, 0.95, "(d)", transform=axd.transAxes, fontsize=10,
             fontweight="bold", va="top")

    fig.legend(handles=[Line2D([], [], color="0.85", lw=5, label="Fokker--Planck")]
               + [Line2D([], [], color=COL[m], lw=LW[m], label=LBL[m])
                  for m in present_plot],
               loc="lower center", bbox_to_anchor=(0.53, -0.012), ncol=4,
               columnspacing=1.8, handlelength=1.9)
    # run parameters belong in the LaTeX caption, not on the canvas

    out = os.path.join(args.outdir, "fig_doublewell")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    print(f"Wrote {out}.pdf / .png")
