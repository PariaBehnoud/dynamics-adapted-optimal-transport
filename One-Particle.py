import time
import platform
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy.integrate import solve_ivp

# ==============================================================================
# 1. PARAMETERS  (exactly the values quoted in the manuscript)
# ==============================================================================
k, D = 1.0, 0.25            # spring constant, diffusion coefficient
F0, omega = 2.0, 1.0        # force amplitude, cyclic frequency
t_s = 2.0                   # shock (step) onset
t_0, tau_r = 2.0, 2.0       # ramp onset, ramp rise time
m0, var0 = 1.5, 0.05        # <x>_{P(0)}, Var_{P(0)}
tau, T = 0.05, 6.0          # JKO time step, total horizon for Table I
M, x_min, x_max = 641, -6.0, 6.0        # grid
eps = 0.05                  # entropic regularization, main runs
T_ref = 4.0                 # horizon for the refinement studies
eps_tau = 0.2               # eps held fixed in the tau-refinement study
N_SINK, TOL = 800, 1e-12    # Sinkhorn cap and relative tolerance

# Langevin benchmark
N_TRAJ, N_SUB, SEED = 200_000, 20, 0


def gamma_eps(e):
    """Diffusion-consistent entropy coefficient: eps + 2*gamma = 1."""
    e = float(e)
    if not 0.0 <= e <= 1.0:
        raise ValueError("require 0 <= eps <= 1")
    return 0.5 * (1.0 - e)


# ==============================================================================
# 2. FORCE PROTOCOLS
# ==============================================================================
def make_cyclic(A, w):
    return lambda t: A * np.cos(w * t)


def make_step(A, ts):
    return lambda t: A * (1.0 if t >= ts else 0.0)


def make_ramp(A, t0, tr):
    def F(t):
        if t < t0:
            return 0.0
        if t < t0 + tr:
            return A * (t - t0) / tr
        return A
    return F


protocols = {
    "step":   make_step(F0, t_s),
    "ramp":   make_ramp(F0, t_0, tau_r),
    "cyclic": make_cyclic(F0, omega),
}
LABEL = {"step": "Step", "ramp": "Ramp", "cyclic": "Cyclic"}


# ==============================================================================
# 3. EXACT OU MOMENTS AND DETERMINISTIC PROTOCOL SHIFT eta_p
# ==============================================================================
# max_step caps the step size so the integrator cannot stride over the
# discontinuity of the step protocol at t = t_s or the kinks of the ramp.
def _ivp(rhs, y0, t_grid):
    return solve_ivp(rhs, (t_grid[0], t_grid[-1]), y0, t_eval=t_grid,
                     rtol=1e-12, atol=1e-14, method="DOP853",
                     max_step=0.01).y


def exact_ou(F, t_grid):
    """Exact OU mean and variance, integrated to machine precision."""
    y = _ivp(lambda t, y: [-k * y[0] + F(t), -2.0 * k * y[1] + 2.0 * D],
             [m0, var0], t_grid)
    return y[0], y[1]


def eta_p(F, t_grid):
    """eta_p' = -k eta_p + F(t),  eta_p(0) = 0."""
    return _ivp(lambda t, y: [-k * y[0] + F(t)], [0.0], t_grid)[0]


# ==============================================================================
# 4. GENERALIZED SINKHORN JKO SOLVER
# ==============================================================================
def jko_1d(x, p0_density, F, tau, n_steps, eps, gamma=None,
           n_sink=N_SINK, tol=TOL):
    """One-dimensional regularized JKO with the exact OU endpoint cost.

    gamma=None  ->  the diffusion-consistent choice (1 - eps)/2.
    Pass gamma=0.5 explicitly to reproduce the inconsistent fixed-gamma scheme.
    Returns (t, P, mean Sinkhorn iterations per step).
    """
    if eps <= 0.0:
        raise ValueError("the Sinkhorn iteration requires eps > 0")
    gamma = gamma_eps(eps) if gamma is None else float(gamma)
    lam = gamma / eps                       # exponent 1/(1 + gamma/eps)
    dx = x[1] - x[0]

    t = np.arange(n_steps + 1) * tau
    ep = eta_p(F, t)
    Sigma_tau = (D / k) * (1.0 - np.exp(-2.0 * k * tau))
    decay = np.exp(-k * tau)

    p = p0_density * dx
    p = p / p.sum()                          # probability masses on the grid
    P = np.empty((n_steps + 1, x.size))
    P[0] = p / dx
    iters = []

    for j in range(n_steps):
        # exact deterministic one-step flow map of the driven harmonic system
        Phi = ep[j + 1] + decay * (x - ep[j])
        C = (x[None, :] - Phi[:, None]) ** 2 / (2.0 * Sigma_tau)
        C = C - C.min(axis=1, keepdims=True)    # row shift: does not change OT
        Kmat = np.exp(-C / eps)

        b = np.ones_like(x)
        for it in range(n_sink):
            a = p / np.maximum(Kmat @ b, 1e-300)
            z = np.maximum(Kmat.T @ a, 1e-300)
            q = z ** (1.0 / (1.0 + lam))
            q /= q.sum()
            b_new = q / z
            converged = (np.max(np.abs(b_new - b))
                         < tol * max(1.0, np.max(np.abs(b))))
            b = b_new
            if converged:
                break
        iters.append(it + 1)
        p = q
        P[j + 1] = p / dx

    return t, P, float(np.mean(iters))


def ou_markov_propagation(x, p0_density, F, tau, n_steps):
    """The eps = 1, gamma = 0 endpoint, evaluated directly: push the density
    through the row-normalized OU transition kernel, with no Sinkhorn."""
    dx = x[1] - x[0]
    t = np.arange(n_steps + 1) * tau
    ep = eta_p(F, t)
    Sigma_tau = (D / k) * (1.0 - np.exp(-2.0 * k * tau))
    decay = np.exp(-k * tau)
    p = p0_density * dx
    p = p / p.sum()
    P = np.empty((n_steps + 1, x.size))
    P[0] = p / dx
    for j in range(n_steps):
        Phi = ep[j + 1] + decay * (x - ep[j])
        C = (x[None, :] - Phi[:, None]) ** 2 / (2.0 * Sigma_tau)
        Kmat = np.exp(-(C - C.min(axis=1, keepdims=True)))
        p = p @ (Kmat / Kmat.sum(axis=1, keepdims=True))
        p /= p.sum()
        P[j + 1] = p / dx
    return t, P


# ==============================================================================
# 5. FINITE-TAU GAUSSIAN REDUCTION  (Appendix: regularized variance recursion)
#      w = u - gamma S  solves   w^2 - (A + eps S) w - A gamma S = 0
#      u = gamma S + (1/2)[ (A + eps S) + sqrt( (A + eps S)^2 + 4 A gamma S ) ]
# ==============================================================================
def gaussian_reduction(F, tau, n_steps, eps, gamma=None):
    gamma = gamma_eps(eps) if gamma is None else float(gamma)
    t = np.arange(n_steps + 1) * tau
    ep = eta_p(F, t)
    S = (D / k) * (1.0 - np.exp(-2.0 * k * tau))
    a = np.exp(-k * tau)
    m = np.empty(n_steps + 1)
    v = np.empty(n_steps + 1)
    m[0], v[0] = m0, var0
    for j in range(n_steps):
        m[j + 1] = ep[j + 1] + a * (m[j] - ep[j])       # exact at every tau
        A = a * a * v[j]
        w = 0.5 * ((A + eps * S)
                   + np.sqrt((A + eps * S) ** 2 + 4.0 * A * gamma * S))
        v[j + 1] = w + gamma * S
    return t, m, v


# ==============================================================================
# 6. DIAGNOSTICS
# ==============================================================================
def moments(x, P):
    dx = x[1] - x[0]
    mean = (x[None, :] * P).sum(axis=1) * dx
    var = ((x[None, :] - mean[:, None]) ** 2 * P).sum(axis=1) * dx
    return mean, var


def w1_distance(x, P, Q):
    r"""W_1(P,Q) = \int |F_P(x) - F_Q(x)| dx  in one dimension,
    with F the cumulative distribution function (trapezoid-free Riemann sum
    on the uniform grid). Note this uses the ground metric |x - y|, distinct
    from the endpoint cost c_tau used inside the Sinkhorn step."""
    dx = x[1] - x[0]
    return float(np.abs(np.cumsum((P - Q) * dx)).sum() * dx)


def gaussian_density(x, mean, var):
    return (np.exp(-(x[None, :] - mean[:, None]) ** 2 / (2.0 * var[:, None]))
            / np.sqrt(2.0 * np.pi * var[:, None]))


def max_var_error(x, P, F, t):
    _, v = moments(x, P)
    _, ve = exact_ou(F, t)
    return float(np.max(np.abs(v - ve)))


# ==============================================================================
# 7. GRID, INITIAL DENSITY
# ==============================================================================
x = np.linspace(x_min, x_max, M)
p0 = np.exp(-(x - m0) ** 2 / (2.0 * var0)) / np.sqrt(2.0 * np.pi * var0)
n_steps = int(round(T / tau))
F_cyc = protocols["cyclic"]
g_main = gamma_eps(eps)

print("=" * 79)
print("ONE-BEAD VALIDATION OF THE DIFFUSION-CONSISTENT REGULARIZED JKO SCHEME")
print("=" * 79)
print(f"k = {k}, D = {D}, F0 = {F0}, omega = {omega}, t_s = {t_s}, "
      f"t_0 = {t_0}, tau_r = {tau_r}")
print(f"<x>_P(0) = {m0}, Var_P(0) = {var0}, x in [{x_min}, {x_max}], M = {M}")
print(f"tau = {tau}, eps = {eps}, gamma = (1-eps)/2 = {g_main:g}, "
      f"eps + 2 gamma = {eps + 2 * g_main:.1f}")
print(f"Sinkhorn: tol = {TOL:g} (relative), max {N_SINK} iterations")
print(f"platform: {platform.platform()} | numpy {np.__version__}")

# ==============================================================================
# [1] TABLE I -- three protocols, max errors over t in [0, T]
# ==============================================================================
results, table1 = {}, []
for name, F in protocols.items():
    t_0w = time.perf_counter()
    t, P, avg_it = jko_1d(x, p0, F, tau, n_steps, eps)
    wall = time.perf_counter() - t_0w

    mj, vj = moments(x, P)
    me, ve = exact_ou(F, t)
    Pe = gaussian_density(x, me, ve)
    W1 = np.array([w1_distance(x, P[j], Pe[j]) for j in range(n_steps + 1)])

    table1.append([name, np.max(np.abs(mj - me)), np.max(np.abs(vj - ve)),
                   W1.max(), avg_it, wall])
    results[name] = dict(t=t, mean=mj, var=vj, mean_e=me, var_e=ve,
                         eta=eta_p(F, t), W1=W1)

print("\n" + "-" * 79)
print(f"[1] TABLE I   max errors over t in [0,{T:g}]   "
      f"(tau = {tau}, eps = {eps}, gamma = {g_main:g}, M = {M})")
print("-" * 79)
print(f"{'protocol':<9}{'max|<x>_P - m|':>17}{'max|Var_P - Sigma|':>21}"
      f"{'max W1':>11}{'Sinkhorn it/step':>19}{'wall [s]':>10}")
for r in table1:
    print(f"{r[0]:<9}{r[1]:>17.2e}{r[2]:>21.3e}{r[3]:>11.3e}"
          f"{r[4]:>19.1f}{r[5]:>10.1f}")

# ==============================================================================
# [2] EPS TABLE -- consistent gamma vs gamma = 1/2, at fixed tau
# ==============================================================================
eps_list = np.array([0.025, 0.05, 0.10, 0.20, 0.50, 1.00])
n_ref = int(round(T_ref / tau))
t_ref = np.arange(n_ref + 1) * tau
eps_rows = []
for e in eps_list:
    _, P_c, it_c = jko_1d(x, p0, F_cyc, tau, n_ref, e)                  # gamma_eps
    _, P_o, _ = jko_1d(x, p0, F_cyc, tau, n_ref, e, gamma=0.5)          # gamma=1/2
    _, v_c = moments(x, P_c)
    _, ve = exact_ou(F_cyc, t_ref)
    _, _, v_g = gaussian_reduction(F_cyc, tau, n_ref, e)
    eps_rows.append([e, gamma_eps(e),
                     float(np.max(np.abs(v_c - ve))),
                     max_var_error(x, P_o, F_cyc, t_ref),
                     float(np.max(np.abs(v_c - v_g))), it_c])

print("\n" + "-" * 79)
print(f"[2] EPS STUDY   max|Var_P - Sigma| over t in [0,{T_ref:g}]   "
      f"(tau = {tau}, cyclic forcing)")
print("-" * 79)
print(f"{'eps':>7}{'gamma=(1-eps)/2':>18}{'consistent err':>16}"
      f"{'err gamma=1/2':>16}{'Sinkhorn it':>13}")
for r in eps_rows:
    print(f"{r[0]:>7.3f}{r[1]:>18.4f}{r[2]:>16.3e}{r[3]:>16.3e}{r[5]:>13.0f}")

# ==============================================================================
# [3] ENDPOINT CHECKS -- eps = 0 (analytic) and eps = 1 (Markov kernel)
# ==============================================================================
_, _, v_e0 = gaussian_reduction(F_cyc, tau, n_ref, 0.0, gamma=0.5)
_, ve_ref = exact_ou(F_cyc, t_ref)
t_mk, P_mk = ou_markov_propagation(x, p0, F_cyc, tau, n_ref)
_, P_s1, it_s1 = jko_1d(x, p0, F_cyc, tau, n_ref, 1.0)

print("\n" + "-" * 79)
print("[3] ENDPOINT CHECKS")
print("-" * 79)
print(f"eps = 0, gamma = 1/2 (analytic recursion):  "
      f"max|Var - Sigma| = {np.max(np.abs(v_e0 - ve_ref)):.3e}")
print(f"eps = 1, gamma = 0:  max|P_Sinkhorn - P_Markov| = "
      f"{np.max(np.abs(P_s1 - P_mk)):.2e},  "
      f"max|Var - Sigma| = {max_var_error(x, P_mk, F_cyc, t_mk):.2e},  "
      f"Sinkhorn it/step = {it_s1:.0f}")

# ==============================================================================
# [4] TAU TABLE -- time-step refinement at fixed eps, NO bias subtraction
# ==============================================================================
tau_list = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
err_new, err_old = [], []
for tt in tau_list:
    n = int(round(T_ref / tt))
    t_c, P_c, _ = jko_1d(x, p0, F_cyc, tt, n, eps_tau)
    _, P_o, _ = jko_1d(x, p0, F_cyc, tt, n, eps_tau, gamma=0.5)
    err_new.append(max_var_error(x, P_c, F_cyc, t_c))
    err_old.append(max_var_error(x, P_o, F_cyc, t_c))
err_new, err_old = np.array(err_new), np.array(err_old)

print("\n" + "-" * 79)
print(f"[4] TAU REFINEMENT   eps = {eps_tau} fixed, t in [0,{T_ref:g}], "
      f"no regularization-bias subtraction")
print("-" * 79)
print(f"{'tau':>9}{'gamma=(1-eps)/2':>18}{'gamma=1/2':>14}{'ratio':>9}")
for i, tt in enumerate(tau_list):
    ratio = err_new[i - 1] / err_new[i] if i else np.nan
    rs = "     --- " if i == 0 else f"{ratio:>9.2f}"
    print(f"{tt:>9.4f}{err_new[i]:>18.3e}{err_old[i]:>14.3e}{rs}")
print(f"saturation level of the gamma = 1/2 scheme:  "
      f"(D/k) eps = {(D / k) * eps_tau:.3f}")

# ==============================================================================
# [5] GAUSSIAN-RECURSION AGREEMENT for the main run
# ==============================================================================
t_g, _, var_g = gaussian_reduction(F_cyc, tau, n_steps, eps)
print("\n" + "-" * 79)
print("[5] GRID SINKHORN vs FINITE-TAU GAUSSIAN RECURSION")
print("-" * 79)
print(f"max_t |Var_grid - Var_Gauss| = "
      f"{np.max(np.abs(results['cyclic']['var'] - var_g)):.2e}"
      f"   (max over the eps study: "
      f"{max(r[4] for r in eps_rows):.1e})")

# ==============================================================================
# [6] LANGEVIN BENCHMARK (Euler-Maruyama, cyclic forcing)
# ==============================================================================
rng = np.random.default_rng(SEED)
t_0w = time.perf_counter()
X = m0 + np.sqrt(var0) * rng.standard_normal(N_TRAJ)
dt = tau / N_SUB
lm, lv = [X.mean()], [X.var()]
for j in range(n_steps):
    for s in range(N_SUB):
        tt = j * tau + s * dt
        X += ((-k * X + F_cyc(tt)) * dt
              + np.sqrt(2.0 * D * dt) * rng.standard_normal(N_TRAJ))
    lm.append(X.mean())
    lv.append(X.var())
wall_lang = time.perf_counter() - t_0w
rc = results["cyclic"]
lang_m = float(np.max(np.abs(np.array(lm) - rc["mean_e"])))
lang_v = float(np.max(np.abs(np.array(lv) - rc["var_e"])))

print("\n" + "-" * 79)
print(f"[6] LANGEVIN (Euler-Maruyama, {N_TRAJ} trajectories, "
      f"{N_SUB} substeps/interval, seed {SEED})")
print("-" * 79)
print(f"max|mean err| = {lang_m:.2e}   max|var err| = {lang_v:.2e}   "
      f"wall = {wall_lang:.1f} s")

# ==============================================================================
# [7] TIMINGS -- numbers for the timing paragraph
# ==============================================================================
wall_jko = float(np.mean([r[5] for r in table1]))
print("\n" + "-" * 79)
print("[7] WALL-CLOCK SUMMARY  (for the timing paragraph)")
print("-" * 79)
print(f"generalized JKO, {n_steps} steps, M = {M}, eps = {eps}: "
      f"{wall_jko:.1f} s per protocol "
      f"({1e3 * wall_jko / n_steps:.0f} ms per JKO step)")
print(f"Langevin, {N_TRAJ} trajectories x {N_SUB} substeps: {wall_lang:.1f} s")
print(f"ratio JKO/Langevin: {wall_jko / wall_lang:.2f}"
      "   (in 1D the dense M x M Sinkhorn step is NOT cheaper than "
      "Euler-Maruyama;\n    the advantage of the transport formulation here is "
      "accuracy at fixed tau, not speed)")

# ==============================================================================
# 8. FIGURE 2  (APS two-column width, Okabe-Ito colour-blind-safe palette)
# ==============================================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "legend.handlelength": 2.2,
    "savefig.dpi": 600, "pdf.fonttype": 42,
})
COL = {"step": "#0072B2", "ramp": "#E69F00", "cyclic": "#009E73"}
C_NEW, C_OLD = "#0072B2", "#D55E00"
EXACT = dict(color="k", ls=(0, (4, 2.5)), lw=0.9)

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1))
(ax1, ax2), (ax3, ax4) = axes


def panel(ax, s):
    ax.text(0.025, 0.965, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


# ---- (a) mean response -------------------------------------------------------
for name in protocols:
    r = results[name]
    ax1.plot(r["t"], r["mean"], color=COL[name], lw=1.8)
    ax1.plot(r["t"], r["mean_e"], **EXACT)
ax1.axvline(t_s, color="0.75", lw=0.6, zorder=0)
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$\langle x\rangle_{P(t)}$")
ax1.set_xlim(0, T)
h = [Line2D([], [], color=COL[n], lw=1.8, label=LABEL[n]) for n in protocols]
h.append(Line2D([], [], label="Exact OU", **EXACT))
ax1.legend(handles=h, loc="lower left", ncol=2, columnspacing=1.2)
panel(ax1, "(a)")

# ---- (b) tracking error -----------------------------------------------------
STY = {"step": dict(lw=2.4, ls="-"),
       "ramp": dict(lw=1.6, ls=(0, (5, 2))),
       "cyclic": dict(lw=1.1, ls=(0, (1.2, 1.2)))}
for name in protocols:
    r = results[name]
    ax2.semilogy(r["t"], np.abs(r["mean"] - r["eta"]), color=COL[name],
                 label=LABEL[name], **STY[name])
t_b = results["cyclic"]["t"]
d0 = abs(m0 - results["cyclic"]["eta"][0])          # |Delta(0)|
ax2.semilogy(t_b[::12], d0 * np.exp(-k * t_b[::12]), "o", ms=3.5, mfc="none",
             mec="k", mew=0.7, label=r"$|\Delta(0)|\,e^{-kt}$")
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$|\Delta(t)| = |\langle x\rangle_{P(t)} - \eta_p(t)|$")
ax2.set_xlim(0, T)
ax2.set_ylim(2.5e-3, 6.0)
ax2.legend(loc="upper right")
panel(ax2, "(b)")

# ---- (c) variance, with zoom inset ------------------------------------------
# the three protocols give identical variance, so draw them as nested bands
VLW = {"step": 3.4, "ramp": 2.0, "cyclic": 0.9}
for name in ("step", "ramp", "cyclic"):
    ax3.plot(results[name]["t"], results[name]["var"], color=COL[name],
             lw=VLW[name], solid_capstyle="butt")
ax3.plot(rc["t"], rc["var_e"], **EXACT)
ax3.axhline(D / k, color="0.6", lw=0.6, ls=":", zorder=0)
ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$\mathrm{Var}_{P(t)}[x]$")
ax3.set_xlim(0, T)
ax3.set_ylim(0.03, 0.285)
vh = [Line2D([], [], color=COL[n], lw=VLW[n], label=LABEL[n])
      for n in ("step", "ramp", "cyclic")]
vh.append(Line2D([], [], label="Exact OU", **EXACT))
ax3.legend(handles=vh, loc="center right", bbox_to_anchor=(1.01, 0.66),
           ncol=2, columnspacing=1.1, handlelength=1.8,
           title="JKO (curves coincide)", title_fontsize=7.5)
panel(ax3, "(c)")

ins = ax3.inset_axes([0.47, 0.08, 0.5, 0.36])
ins.plot(rc["t"], rc["var"], color=COL["cyclic"], lw=1.6)
ins.plot(rc["t"], rc["var_e"], **EXACT)
ins.plot(t_g[::6], var_g[::6], "o", ms=2.8, mfc="none", mec=C_OLD, mew=0.7)
ins.set_xlim(3, 6)
ins.set_ylim(0.241, 0.2515)
ins.tick_params(labelsize=6.5, length=2)
ins.minorticks_off()
ins.set_yticks([0.242, 0.246, 0.250])
ins.set_xticks([3, 4.5, 6])
ins.text(0.5, 0.42, "o  Gaussian recursion", transform=ins.transAxes,
         fontsize=6.3, color=C_OLD, ha="center", va="center")
ins.text(0.5, 0.71, "exact OU", transform=ins.transAxes, fontsize=6.3,
         ha="center", va="center")
ins.set_title(r"zoom, $t\in[3,6]$", fontsize=6.5, pad=2)

# ---- (d) tau refinement -----------------------------------------------------
ax4.loglog(tau_list, err_old, "s--", color=C_OLD, ms=4.5, lw=1.2,
           mfc="white", mew=1.0, label=r"$\gamma = 1/2$")
ax4.loglog(tau_list, err_new, "o-", color=C_NEW, ms=4.5, lw=1.4,
           label=r"$\gamma = (1-\varepsilon)/2$")
ax4.axhline((D / k) * eps_tau, color=C_OLD, lw=0.7, ls=":")
ax4.text(tau_list[-1] * 1.05, (D / k) * eps_tau * 1.12,
         r"$(D/k)\,\varepsilon$", color=C_OLD, fontsize=8)
c1 = err_new[-1] / tau_list[-1]
tt2 = np.array([tau_list[-1], tau_list[0]])
ax4.loglog(tt2, 0.55 * c1 * tt2, color="0.4", lw=0.8)
ax4.text(0.05, 0.55 * c1 * 0.05 * 0.62, r"$\propto\tau$", color="0.3",
         fontsize=8.5, ha="left")
ax4.set_xlabel(r"time step $\tau$")
ax4.set_ylabel(r"$\max_t\,|\mathrm{Var}_{P}-\mathrm{Var}_{\mathrm{OU}}|$")
ax4.set_ylim(8e-4, 1.2e-1)
ax4.yaxis.set_minor_formatter(NullFormatter())
ax4.xaxis.set_minor_formatter(NullFormatter())
ax4.set_xticks(tau_list)
ax4.set_xticklabels(["0.2", "0.1", "0.05", "0.025", "0.0125"])
ax4.set_xlim(0.0105, 0.24)
ax4.legend(loc="lower right", title=rf"$\varepsilon = {eps_tau}$",
           title_fontsize=7.5)
panel(ax4, "(d)")

fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.0)
fig.savefig("fig2_validation.pdf", bbox_inches="tight")
fig.savefig("fig2_validation.png", dpi=600, bbox_inches="tight")

plt.show()

# ==============================================================================
# 9. CSV OUTPUT
# ==============================================================================
np.savetxt("fig2_table1_protocol_errors.csv",
           np.array([r[1:] for r in table1], dtype=float), delimiter=",",
           header="max_mean_err,max_var_err,max_W1,sinkhorn_it_per_step,wall_s"
                  "  (rows: step, ramp, cyclic)", comments="")
np.savetxt("fig2_epsilon_study.csv", np.array(eps_rows), delimiter=",",
           header="eps,gamma_consistent,err_consistent,err_gamma_half,"
                  "grid_minus_gaussian,sinkhorn_it_per_step", comments="")
np.savetxt("fig2_tau_study.csv",
           np.column_stack([tau_list, err_new, err_old]), delimiter=",",
           header=f"tau,err_consistent,err_gamma_half  (eps={eps_tau})",
           comments="")

print("\n" + "-" * 79)
print("[8] FILES WRITTEN")
print("-" * 79)
for f in ("fig2_validation.pdf", "fig2_validation.png",
          "fig2_table1_protocol_errors.csv", "fig2_epsilon_study.csv",
          "fig2_tau_study.csv"):
    print("   " + f)
