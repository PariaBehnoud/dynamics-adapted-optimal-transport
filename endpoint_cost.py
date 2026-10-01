r"""
================================================================================
ACCURACY OF THE APPROXIMATE ENDPOINT KERNELS AGAINST THE EXACT VARIATIONAL COST
Produces the "endpoint-kernel accuracy" panel of the double-well figure.
================================================================================

The exact endpoint cost of Eq. (nonlinear_exact_endpoint_cost) is

    c(x,y) = inf   (1/4D) int_0^tau | eta' + V'(eta) |^2 dt ,
           eta(0)=x, eta(tau)=y

whose Euler-Lagrange equation is obtained by varying the residual action.  With
L = (1/4D)(eta' + V'(eta))^2,

    d/dt [ (1/2D)(eta' + V') ] = (1/2D)(eta' + V') V''
    =>  eta'' + V'' eta' = (eta' + V') V''
    =>  eta'' = V'(eta) V''(eta) = (1/2) d/d(eta) [ V'(eta)^2 ] .

So the minimizing path is a Newtonian trajectory in the effective potential
-V'^2/2.  We solve this two-point boundary-value problem directly and evaluate
the action on the solution.  This is the "independent accuracy check on the
approximate endpoint kernel" referred to in Sec. 9.G -- it is far too expensive
to use inside the time stepping (it needs a BVP solve for every source-target
pair at every step) but it is exactly what is needed to grade the approximations.

Kernels compared (a row-dependent additive constant does not change the optimal
transport problem, so every cost is compared after subtracting its minimum over
the sampled targets, and the error is normalized by the range of the exact cost
over the same targets):

    STA     c^STA        , Eq. (STA_kernel)
    STA+J   c^STA,J      , Eq. (STA_kernel_jacobian)
    JKO     the standard non-autonomous JKO cost, i.e. c^STA without its
            drift term:  |y-x|^2/(4 D tau) + [V(y)-V(x)]/(2D)
    LG      local-Gaussian kernel, Eq. (local_gaussian_cost)
    FM      flow-map-only kernel, Eq. (flow_map_only)

The controlling parameter is tau |V''(x)|: the straight-segment path used by
the STA expansion is a good stand-in for the true minimizer only while the
drift cannot appreciably bend the path over one step.

Usage:
    python3 endpoint_cost_PRE.py                 # both wells, default sweep
    python3 endpoint_cost_PRE.py --a 1.0 --F0 1.8
================================================================================
"""

import argparse
import numpy as np
from scipy.integrate import solve_bvp

ap = argparse.ArgumentParser()
ap.add_argument("--D", type=float, default=0.25)
ap.add_argument("--nx", type=int, default=33, help="source points across the domain")
ap.add_argument("--ny", type=int, default=21, help="targets sampled per source")
ap.add_argument("--nsig", type=float, default=3.0, help="target window in kernel sigma")
ap.add_argument("--xmax", type=float, default=2.0)
ap.add_argument("--bvp-tol", type=float, default=1e-10)
ap.add_argument("--prefix", type=str, default="fig_endpoint_cost")
ap.add_argument("--no-figure", action="store_true")
args = ap.parse_args()

D = args.D
TAUS = (0.2, 0.1, 0.05, 0.025)
# (well depth a, force F0, label).  F0 = 1.2 F_c in both cases, so the two
# wells are at the same point of their own force scale and differ only in
# curvature -- which is the variable under test.
WELLS = [(1.0, 1.848, r"deep,  $a=1$"), (0.25, 0.462, r"shallow,  $a=1/4$")]
GLs, GLw = np.polynomial.legendre.leggauss(16)
GLs = 0.5 * (GLs + 1.0)
GLw = 0.5 * GLw


def make_well(a, F0):
    V = lambda z: a * (z ** 2 - 1.0) ** 2 - F0 * z
    dV = lambda z: a * 4.0 * z * (z ** 2 - 1.0) - F0
    d2V = lambda z: a * (12.0 * z ** 2 - 4.0)
    return V, dV, d2V


# ==============================================================================
# EXACT ENDPOINT COST BY BOUNDARY-VALUE SOLVE
# ==============================================================================
def exact_cost(x0, y0, tau, dV, d2V, tol):
    t = np.linspace(0.0, tau, 41)
    guess = np.vstack([x0 + (y0 - x0) * t / tau,
                       np.full_like(t, (y0 - x0) / tau)])
    s = solve_bvp(lambda tt, y: np.vstack([y[1], dV(y[0]) * d2V(y[0])]),
                  lambda ya, yb: np.array([ya[0] - x0, yb[0] - y0]),
                  t, guess, tol=tol, max_nodes=50000)
    if not s.success:
        return np.nan
    tt = np.linspace(0.0, tau, 801)
    z = s.sol(tt)
    return float(np.trapezoid((z[1] + dV(z[0])) ** 2, tt) / (4.0 * D))


# ==============================================================================
# APPROXIMATE KERNELS
# ==============================================================================
def flow_map(x0, tau, dV, nsub=200):
    z, h = float(x0), tau / nsub
    for _ in range(nsub):
        k1 = -dV(z); k2 = -dV(z + 0.5 * h * k1)
        k3 = -dV(z + 0.5 * h * k2); k4 = -dV(z + h * k3)
        z += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def approx_costs(x0, ys, tau, V, dV, d2V):
    z = x0 + GLs[:, None] * (ys[None, :] - x0)
    q2 = np.sum(GLw[:, None] * dV(z) ** 2, axis=0)
    lap = np.sum(GLw[:, None] * d2V(z), axis=0)
    quad = (ys - x0) ** 2 / (4.0 * D * tau)
    pot = (V(ys) - V(x0)) / (2.0 * D)
    sta = quad + pot + tau / (4.0 * D) * q2
    Phi = flow_map(x0, tau, dV)
    H = d2V(x0)
    Sig = 2.0 * D * tau if abs(H) < 1e-10 else D * (1.0 - np.exp(-2.0 * H * tau)) / H
    return {"STA": sta,
            "STA+J": sta - 0.5 * tau * lap,
            "JKO": quad + pot,
            "LG": (ys - Phi) ** 2 / (2.0 * Sig),
            "FM": (ys - Phi) ** 2 / (4.0 * D * tau)}


NAMES = ["STA", "STA+J", "JKO", "LG", "FM"]


# ==============================================================================
# SWEEP
# ==============================================================================
def sweep(a, F0):
    V, dV, d2V = make_well(a, F0)
    xs = np.linspace(-args.xmax, args.xmax, args.nx)
    rows = []
    for tau in TAUS:
        w = args.nsig * np.sqrt(2.0 * D * tau)
        for x0 in xs:
            centre = x0 - tau * dV(x0)               # Euler step
            ys = np.linspace(centre - w, centre + w, args.ny)
            ce = np.array([exact_cost(x0, y, tau, dV, d2V, args.bvp_tol) for y in ys])
            ok = np.isfinite(ce)
            if ok.sum() < args.ny // 2:
                continue
            ce = ce[ok] - ce[ok].min()
            rng = max(np.ptp(ce), 1e-30)
            ca = approx_costs(x0, ys, tau, V, dV, d2V)
            rows.append({"tau": tau, "x": x0, "tvpp": tau * abs(d2V(x0)),
                         **{n: float(np.max(np.abs((ca[n][ok] - ca[n][ok].min()) - ce)) / rng)
                            for n in NAMES}})
    return xs, rows


print("=" * 79)
print("ENDPOINT-KERNEL ACCURACY AGAINST THE EXACT VARIATIONAL COST")
print("=" * 79)
print(f"D = {D};  exact cost from the BVP  eta'' = V'(eta) V''(eta),  "
      f"tol = {args.bvp_tol:g}")
print(f"{args.nx} source points on [{-args.xmax}, {args.xmax}], {args.ny} targets each, "
      f"tau in {TAUS}")
print("error = max_y |c_approx - c_exact| / range(c_exact), after removing the "
      "row constant\n")

ALL = {}
for a, F0, lab in WELLS:
    xs, rows = sweep(a, F0)
    ALL[lab] = rows
    print("-" * 79)
    print(f"{lab.replace('$','').replace(chr(92)+'  ',' ')}   F0 = {F0}  "
          f"(= 1.2 F_c)")
    print("-" * 79)
    print(f"{'tau':>8}{'max tau|Vpp|':>14}" + "".join(f"{n:>11}" for n in NAMES))
    for tau in TAUS:
        sub = [r for r in rows if r["tau"] == tau]
        print(f"{tau:>8.4f}{max(r['tvpp'] for r in sub):>14.2f}"
              + "".join(f"{max(r[n] for r in sub):>11.4f}" for n in NAMES))

# ------------------------------------------------------------------
# the collapse: error against tau |V''(x)|
# ------------------------------------------------------------------
print("\n" + "-" * 79)
print("COLLAPSE: STA error binned by tau |V''(x)|, both wells and all tau pooled")
print("-" * 79)
pool = [r for rows in ALL.values() for r in rows]
edges = [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 1e9]
print(f"{'tau|Vpp| bin':>16}{'n':>6}" + "".join(f"{n:>11}" for n in NAMES))
for lo, hi in zip(edges[:-1], edges[1:]):
    sub = [r for r in pool if lo <= r["tvpp"] < hi]
    if not sub:
        continue
    lab = f"[{lo:g}, {hi:g})" if hi < 1e8 else f">= {lo:g}"
    print(f"{lab:>16}{len(sub):>6}"
          + "".join(f"{np.median([r[n] for r in sub]):>11.4f}" for n in NAMES))
print("(medians; the STA column is the validity criterion made quantitative)")

np.savetxt(f"{args.prefix}.csv",
           np.array([[i, r["tau"], r["x"], r["tvpp"]] + [r[n] for n in NAMES]
                     for i, rows in enumerate(ALL.values()) for r in rows]),
           delimiter=",", comments="",
           header="well_index,tau,x,tau_absVpp," + ",".join(NAMES))

if args.no_figure:
    raise SystemExit(0)

# ==============================================================================
# FIGURE
# ==============================================================================
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42,
})
COL = {"STA": "#0072B2", "STA+J": "#56B4E9", "JKO": "#D55E00",
       "LG": "#009E73", "FM": "#CC79A7"}
MK = {"STA": "o", "STA+J": "^", "JKO": "s", "LG": "D", "FM": "v"}

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
ax0, ax1, ax2 = axes


def panel(ax, s):
    ax.text(0.04, 0.95, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")


# (a) cost profiles at one representative source point, deep well
a, F0, _ = WELLS[0]
V, dV, d2V = make_well(a, F0)
x0, tau = 1.18, 0.1
w = args.nsig * np.sqrt(2.0 * D * tau)
ys = np.linspace(x0 - tau * dV(x0) - w, x0 - tau * dV(x0) + w, 41)
ce = np.array([exact_cost(x0, y, tau, dV, d2V, args.bvp_tol) for y in ys])
ca = approx_costs(x0, ys, tau, V, dV, d2V)
ax0.plot(ys, ce - np.nanmin(ce), "k-", lw=2.2, alpha=0.35, label="exact (BVP)")
for n in ("STA", "JKO", "LG"):
    ax0.plot(ys, ca[n] - ca[n].min(), color=COL[n], lw=1.3, label=n)
ax0.set_xlabel(r"target $y$")
ax0.set_ylabel(r"$c(x,y) - \min_y c$")
ax0.set_title(rf"$x={x0}$, $\tau={tau}$, $\tau|V''|={tau*abs(d2V(x0)):.2f}$",
              fontsize=8, pad=3)
ax0.legend(loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.6)
panel(ax0, "(a)")

# (b) error across the domain, deep well, two tau
rows = ALL[WELLS[0][2]]
for tau, ls in ((0.1, "-"), (0.025, (0, (4, 2)))):
    sub = sorted([r for r in rows if r["tau"] == tau], key=lambda r: r["x"])
    xx = [r["x"] for r in sub]
    for n in ("STA", "JKO"):
        ax1.semilogy(xx, [max(r[n], 1e-8) for r in sub], ls=ls, color=COL[n], lw=1.4)
ax1.set_xlabel(r"source $x$")
ax1.set_ylabel("relative error in $c$")
ax1.set_title(r"deep well, $a=1$", fontsize=8, pad=3)
ax1.legend(handles=[Line2D([], [], color=COL["STA"], lw=1.4, label=r"$c^{\rm STA}$"),
                    Line2D([], [], color=COL["JKO"], lw=1.4, label="standard JKO cost"),
                    Line2D([], [], color="0.4", lw=1.4, ls="-", label=r"$\tau=0.1$"),
                    Line2D([], [], color="0.4", lw=1.4, ls=(0, (4, 2)), label=r"$\tau=0.025$")],
           loc="lower center", ncol=2, columnspacing=1.0, handlelength=1.8)
panel(ax1, "(b)")

# (c) the collapse against tau |V''|
for n in ("STA", "STA+J", "JKO", "LG", "FM"):
    tv = np.array([r["tvpp"] for r in pool])
    er = np.array([max(r[n], 1e-9) for r in pool])
    o = np.argsort(tv)
    ax2.loglog(tv[o], er[o], MK[n], ms=2.2, mfc="none", mew=0.55,
               color=COL[n], alpha=0.75, label=n)
ax2.axvline(1.0, color="0.55", lw=0.8, ls=":")
ax2.annotate(r"$\tau|V''|=1$", xy=(1.0, 2.0), xytext=(0.62, 2.0), fontsize=7,
             color="0.35", ha="right", va="center")
ax2.set_xlabel(r"$\tau\,|V''(x)|$")
ax2.set_ylabel("relative error in $c$")
ax2.set_ylim(1e-7, 6.0)
ax2.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02), ncol=3,
           columnspacing=0.7, handlelength=1.0, handletextpad=0.3,
           borderaxespad=0.3)
panel(ax2, "(c)")

fig.tight_layout(pad=0.4, w_pad=1.3)
fig.savefig(f"{args.prefix}.pdf", bbox_inches="tight")
fig.savefig(f"{args.prefix}.png", dpi=600, bbox_inches="tight")
print(f"\nSaved {args.prefix}.pdf / .png / .csv")
