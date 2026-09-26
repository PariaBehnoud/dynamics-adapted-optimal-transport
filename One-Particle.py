import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# PARAMETERS USED IN THE VALIDATION
# ============================================================

k = 1.0
D = 0.25

F0 = 2.0
omega = 1.0

t_step = 2.0
t_ramp_start = 2.0
tau_ramp = 2.0

m0 = 1.5
var0 = 0.05

tau = 0.05
T = 6.0

M = 641
eps = 0.05

x_min = -6.0
x_max = 6.0

gamma = 0.5

# ============================================================
# FORCE PROTOCOLS
# ============================================================


def make_cyclic(F0, omega):
    return lambda t: F0 * np.cos(omega * t)


def make_step(F0, ts):
    return lambda t: F0 * (1.0 if t >= ts else 0.0)


def make_ramp(F0, t0, tau_r):
    def F(t):
        if t < t0:
            return 0.0

        if t < t0 + tau_r:
            return F0 * (t - t0) / tau_r

        return F0

    return F


protocols = {
    "step": make_step(F0, t_step),
    "ramp": make_ramp(F0, t_ramp_start, tau_ramp),
    "cyclic": make_cyclic(F0, omega),
}


# ============================================================
# EXACT ORNSTEIN-UHLENBECK SOLUTION
#
# dX = (-k X + F(t)) dt + sqrt(2D) dW
#
# dm/dt   = -k m + F(t)
# dVar/dt = -2k Var + 2D
# ============================================================


def exact_ou(k, D, F, t_grid, m0, var0):

    def rhs(t, y):
        m = y[0]
        var = y[1]

        dm = -k * m + F(t)
        dvar = -2.0 * k * var + 2.0 * D

        return [dm, dvar]

    sol = solve_ivp(
        rhs,
        (t_grid[0], t_grid[-1]),
        [m0, var0],
        t_eval=t_grid,
        rtol=1e-12,
        atol=1e-14,
        method="DOP853",
    )

    return sol.y[0], sol.y[1]


# ============================================================
# DETERMINISTIC PROTOCOL SHIFT eta_p
#
# eta_p' = -k eta_p + F(t)
# eta_p(0) = 0
# ============================================================


def eta_p(k, F, t_grid):

    def rhs(t, y):
        return [-k * y[0] + F(t)]

    sol = solve_ivp(
        rhs,
        (t_grid[0], t_grid[-1]),
        [0.0],
        t_eval=t_grid,
        rtol=1e-12,
        atol=1e-14,
        method="DOP853",
    )

    return sol.y[0]


# ============================================================
# GENERALIZED-SINKHORN JKO SOLVER
#
# P^{k+1} =
# argmin_P T_ck(P^k,P) + gamma int P log P
#
# gamma = 1/2
# ============================================================


def jko_1d(
    x,
    p0_density,
    k,
    D,
    F,
    tau,
    n_steps,
    eps,
    n_sink=800,
    tol=1e-12,
):

    dx = x[1] - x[0]

    lam = gamma / eps

    t = np.arange(n_steps + 1) * tau

    ep = eta_p(k, F, t)

    # Exact one-step OU covariance
    Sigma_tau = (D / k) * (1.0 - np.exp(-2.0 * k * tau))

    decay = np.exp(-k * tau)

    # Convert density to probability masses
    p = p0_density * dx
    p = p / p.sum()

    P = np.empty((n_steps + 1, len(x)))
    P[0] = p / dx

    sinkhorn_iterations = []

    for j in range(n_steps):

        # ----------------------------------------------------
        # Exact deterministic one-step flow map
        # ----------------------------------------------------

        Phi = (
            ep[j + 1]
            + decay * (x - ep[j])
        )

        # ----------------------------------------------------
        # Exact quadratic endpoint cost
        #
        # c(x,y) =
        # (y-Phi(x))^2 / (2 Sigma_tau)
        # ----------------------------------------------------

        C = (
            x[None, :]
            - Phi[:, None]
        ) ** 2 / (2.0 * Sigma_tau)

        # A row-dependent additive constant does not affect OT
        C -= C.min(axis=1, keepdims=True)

        # Entropic Gibbs kernel
        Kmat = np.exp(-C / eps)

        # ----------------------------------------------------
        # Generalized Sinkhorn fixed point
        # ----------------------------------------------------

        b = np.ones_like(x)

        for it in range(n_sink):

            a = p / np.maximum(
                Kmat @ b,
                1e-300
            )

            z = Kmat.T @ a

            q = np.maximum(
                z,
                1e-300
            ) ** (1.0 / (1.0 + lam))

            q /= q.sum()

            b_new = q / np.maximum(
                z,
                1e-300
            )

            residual = np.max(
                np.abs(b_new - b)
            )

            scale = max(
                1.0,
                np.max(np.abs(b))
            )

            if residual < tol * scale:
                b = b_new
                break

            b = b_new

        sinkhorn_iterations.append(it + 1)

        p = q
        P[j + 1] = p / dx

    return (
        t,
        P,
        np.mean(sinkhorn_iterations),
    )


# ============================================================
# MOMENTS
# ============================================================


def moments(x, P):

    dx = x[1] - x[0]

    mean = (
        x[None, :] * P
    ).sum(axis=1) * dx

    variance = (
        (
            x[None, :]
            - mean[:, None]
        ) ** 2
        * P
    ).sum(axis=1) * dx

    return mean, variance


# ============================================================
# CLOSED-FORM GAUSSIAN REDUCTION OF THE JKO SCHEME
# ============================================================


def gaussian_reduction(
    k,
    D,
    F,
    tau,
    n_steps,
    m0,
    var0,
):

    t = np.arange(
        n_steps + 1
    ) * tau

    ep = eta_p(
        k,
        F,
        t
    )

    S = (
        D / k
    ) * (
        1.0
        - np.exp(-2.0 * k * tau)
    )

    m = np.empty(n_steps + 1)
    var = np.empty(n_steps + 1)

    m[0] = m0
    var[0] = var0

    for j in range(n_steps):

        # Exact finite-tau mean update
        m[j + 1] = (
            ep[j + 1]
            + np.exp(-k * tau)
            * (m[j] - ep[j])
        )

        A = (
            np.exp(-2.0 * k * tau)
            * var[j]
        )

        var[j + 1] = 0.5 * (
            (S + A)
            + np.sqrt(
                (S + A) ** 2
                - S ** 2
            )
        )

    return t, m, var


# ============================================================
# 1D WASSERSTEIN-1 DISTANCE
#
# W1(P,Q) = integral |CDF_P - CDF_Q| dx
# ============================================================


def w1_distance(x, P, Q):

    dx = x[1] - x[0]

    cdf_difference = np.cumsum(
        (P - Q) * dx
    )

    return (
        np.abs(cdf_difference).sum()
        * dx
    )


# ============================================================
# INITIAL DENSITY AND GRID
# ============================================================


x = np.linspace(
    x_min,
    x_max,
    M
)

p0 = np.exp(
    -(x - m0) ** 2
    / (2.0 * var0)
) / np.sqrt(
    2.0 * np.pi * var0
)

n_steps = int(T / tau)


# ============================================================
# RUN STEP, RAMP, CYCLIC PROTOCOLS
# ============================================================


results = {}
table1 = []

for name, F in protocols.items():

    print("\n" + "=" * 60)
    print("Running protocol:", name)
    print("=" * 60)

    t, P, avg_sink = jko_1d(
        x,
        p0,
        k,
        D,
        F,
        tau,
        n_steps,
        eps,
        n_sink=800,
        tol=1e-12,
    )

    mean_jko, var_jko = moments(
        x,
        P
    )

    mean_exact, var_exact = exact_ou(
        k,
        D,
        F,
        t,
        m0,
        var0,
    )

    ep = eta_p(
        k,
        F,
        t
    )

    # Exact Gaussian densities
    P_exact = (
        np.exp(
            -(
                x[None, :]
                - mean_exact[:, None]
            ) ** 2
            / (
                2.0
                * var_exact[:, None]
            )
        )
        /
        np.sqrt(
            2.0
            * np.pi
            * var_exact[:, None]
        )
    )

    W1 = np.array(
        [
            w1_distance(
                x,
                P[j],
                P_exact[j],
            )
            for j in range(n_steps + 1)
        ]
    )

    mean_error = np.max(
        np.abs(
            mean_jko
            - mean_exact
        )
    )

    variance_error = np.max(
        np.abs(
            var_jko
            - var_exact
        )
    )

    w1_error = np.max(W1)

    table1.append(
        [
            name,
            mean_error,
            variance_error,
            w1_error,
        ]
    )

    results[name] = {
        "t": t,
        "P": P,
        "mean_jko": mean_jko,
        "var_jko": var_jko,
        "mean_exact": mean_exact,
        "var_exact": var_exact,
        "eta": ep,
    }

    print(
        "Average Sinkhorn iterations:",
        avg_sink,
    )


# ============================================================
# PRINT TABLE 1
# ============================================================


print("\n")
print("=" * 78)
print("TABLE 1: JKO vs EXACT OU")
print("=" * 78)

print(
    f"{'protocol':<10}"
    f"{'max mean error':>20}"
    f"{'max variance error':>22}"
    f"{'max W1':>15}"
)

for row in table1:

    print(
        f"{row[0]:<10}"
        f"{row[1]:>20.3e}"
        f"{row[2]:>22.3e}"
        f"{row[3]:>15.4e}"
    )


# ============================================================
# EPSILON REFINEMENT STUDY
# ============================================================


eps_list = np.array(
    [
        0.400,
        0.200,
        0.100,
        0.050,
        0.025,
    ]
)

eps_excess_variance = []

# Use cyclic protocol
F_cyclic = protocols["cyclic"]

T_refine = 4.0

for e in eps_list:

    n_ref = int(
        T_refine / tau
    )

    t_eps, P_eps, _ = jko_1d(
        x,
        p0,
        k,
        D,
        F_cyclic,
        tau,
        n_ref,
        eps=e,
        n_sink=800,
        tol=1e-12,
    )

    _, var_eps = moments(
        x,
        P_eps
    )

    _, _, var_gaussian = (
        gaussian_reduction(
            k,
            D,
            F_cyclic,
            tau,
            n_ref,
            m0,
            var0,
        )
    )

    excess = abs(
        var_eps[-1]
        - var_gaussian[-1]
    )

    eps_excess_variance.append(
        excess
    )


eps_excess_variance = np.array(
    eps_excess_variance
)


# ============================================================
# TAU REFINEMENT STUDY
# ============================================================


tau_list = np.array(
    [
        0.200,
        0.100,
        0.050,
        0.025,
    ]
)

tau_variance_error = []

eps_tau_study = 0.05

for tau_test in tau_list:

    n_ref = int(
        round(
            T_refine / tau_test
        )
    )

    t_tau, P_tau, _ = jko_1d(
        x,
        p0,
        k,
        D,
        F_cyclic,
        tau_test,
        n_ref,
        eps=eps_tau_study,
        n_sink=800,
        tol=1e-12,
    )

    _, var_tau = moments(
        x,
        P_tau
    )

    _, var_exact_tau = exact_ou(
        k,
        D,
        F_cyclic,
        t_tau,
        m0,
        var0,
    )

    # Subtract the leading-order entropic regularization contribution (D/k) * epsilon
    error = abs(
        var_tau[-1]
        - var_exact_tau[-1]
        - (D / k) * eps_tau_study
    )

    tau_variance_error.append(
        error
    )


tau_variance_error = np.array(
    tau_variance_error
)


# ============================================================
# PRINT TABLE 2
# ============================================================


print("\n")
print("=" * 78)
print("TABLE 2A: EPSILON REGULARIZATION STUDY")
print("=" * 78)

print(
    f"{'epsilon':>12}"
    f"{'excess variance':>22}"
    f"{'(D/k) epsilon':>22}"
)

for e, measured in zip(
    eps_list,
    eps_excess_variance,
):

    predicted = (
        D / k
    ) * e

    print(
        f"{e:>12.3f}"
        f"{measured:>22.5f}"
        f"{predicted:>22.5f}"
    )


print("\n")
print("=" * 78)
print("TABLE 2B: TAU REFINEMENT STUDY")
print("=" * 78)

print(
    f"{'tau':>12}"
    f"{'variance error':>22}"
)

for tt, error in zip(
    tau_list,
    tau_variance_error,
):

    print(
        f"{tt:>12.3f}"
        f"{error:>22.5e}"
    )


print("\nSuccessive error ratios:")

for i in range(
    len(tau_variance_error) - 1
):

    ratio = (
        tau_variance_error[i]
        / tau_variance_error[i + 1]
    )

    print(
        f"{tau_list[i]:.3f} -> "
        f"{tau_list[i+1]:.3f}: "
        f"{ratio:.3f}"
    )


# ============================================================
# MAKE THE FOUR-PANEL VALIDATION FIGURE
# ============================================================


fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 7.5),
)

ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]


# ------------------------------------------------------------
# PANEL A: MEAN
# ------------------------------------------------------------

for name in [
    "step",
    "ramp",
    "cyclic",
]:

    r = results[name]

    ax1.plot(
        r["t"],
        r["mean_jko"],
        linewidth=2,
        label=f"{name}: JKO",
    )

    ax1.plot(
        r["t"],
        r["mean_exact"],
        "--",
        linewidth=1.2,
    )


ax1.set_xlabel("t")
ax1.set_ylabel(r"$\langle x\rangle$")
ax1.set_title(
    "Mean: JKO vs exact OU"
)

ax1.legend(
    fontsize=8
)


# ------------------------------------------------------------
# PANEL B: TRACKING ERROR
# ------------------------------------------------------------

for name in [
    "step",
    "ramp",
    "cyclic",
]:

    r = results[name]

    delta = np.abs(
        r["mean_jko"]
        - r["eta"]
    )

    ax2.semilogy(
        r["t"],
        delta,
        linewidth=2,
        label=name,
    )


t_plot = results[
    "cyclic"
]["t"]

prediction = (
    abs(m0)
    * np.exp(-k * t_plot)
)

ax2.semilogy(
    t_plot,
    prediction,
    "k:",
    linewidth=2,
    label=r"$|\Delta(0)|e^{-kt}$",
)

ax2.set_xlabel("t")
ax2.set_ylabel(
    r"$|\Delta(t)|$"
)

ax2.set_title(
    r"Tracking error: $|\Delta(t)|$"
)

ax2.legend(
    fontsize=8
)


# ------------------------------------------------------------
# PANEL C: VARIANCE
# ------------------------------------------------------------

for name in [
    "step",
    "ramp",
    "cyclic",
]:

    r = results[name]

    ax3.plot(
        r["t"],
        r["var_jko"],
        linewidth=2,
        label=f"{name}: JKO",
    )

    ax3.plot(
        r["t"],
        r["var_exact"],
        "--",
        linewidth=1.2,
    )


ax3.set_xlabel("t")
ax3.set_ylabel("Variance")

ax3.set_title(
    "Variance: JKO vs exact OU"
)


# ------------------------------------------------------------
# PANEL D: EPS AND TAU STUDIES
# ------------------------------------------------------------

ax4.loglog(
    eps_list,
    eps_excess_variance,
    "o-",
    label=r"$\varepsilon$ study",
)

ax4.loglog(
    eps_list,
    (D / k) * eps_list,
    "k:",
    label=r"$(D/k)\varepsilon$",
)

ax4.loglog(
    tau_list,
    tau_variance_error,
    "s-",
    label=r"$\tau$ study",
)

ax4.set_xlabel(
    r"$\varepsilon$ or $\tau$"
)

ax4.set_ylabel(
    "Variance error"
)

ax4.set_title(
    "Regularization and time-step errors"
)

ax4.legend(
    fontsize=8
)


plt.tight_layout()

plt.savefig(
    "validation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "validation.pdf",
    bbox_inches="tight",
)

plt.show()


# ============================================================
# SAVE TABLES AS CSV FILES
# ============================================================


table1_numeric = np.array(
    [
        [
            row[1],
            row[2],
            row[3],
        ]
        for row in table1
    ]
)

np.savetxt(
    "validation_protocol_errors.csv",
    table1_numeric,
    delimiter=",",
    header=(
        "max_mean_error,"
        "max_variance_error,"
        "max_W1"
    ),
    comments="",
)


eps_table = np.column_stack(
    [
        eps_list,
        eps_excess_variance,
        (D / k) * eps_list,
    ]
)

np.savetxt(
    "validation_epsilon_study.csv",
    eps_table,
    delimiter=",",
    header=(
        "epsilon,"
        "excess_variance,"
        "predicted_D_over_k_times_epsilon"
    ),
    comments="",
)


tau_table = np.column_stack(
    [
        tau_list,
        tau_variance_error,
    ]
)

np.savetxt(
    "validation_tau_study.csv",
    tau_table,
    delimiter=",",
    header="tau,variance_error",
    comments="",
)


print("\nFiles created:")
print("  validation.png")
print("  validation.pdf")
print("  validation_protocol_errors.csv")
print("  validation_epsilon_study.csv")
print("  validation_tau_study.csv")