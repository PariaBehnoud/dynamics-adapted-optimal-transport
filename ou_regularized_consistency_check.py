import numpy as np

# ============================================================
# Regularized-consistency test for the 1D Ornstein-Uhlenbeck case
#
# Model:
#   dX = (-k X + F(t)) dt + sqrt(2D) dW
#
# Regularized variational objective:
#   <C,pi> + eps * Ent(pi) + gamma * Ent(q)
#
# Small-time consistency requires:
#   eps + 2 gamma = 1
#   gamma = (1 - eps)/2
# ============================================================

k = 1.0
D = 0.25
tau = 0.05
T = 6.0
var0 = 0.05
mean0 = 1.5

# Parameters used in the manuscript's cyclic example.
F0 = 2.0
omega = 1.0

def exact_ou_transition_variance(k, D, tau):
    a = np.exp(-k * tau)
    return D / k * (1.0 - a * a)

def regularized_gaussian_variance_step(var_in, k, D, tau, eps, gamma):
    """
    Exact variance update for the Gaussian minimizer of the regularized
    variational problem.

    Let:
        a = exp(-k tau)
        A = a^2 var_in
        S = (D/k)(1-a^2)

    If u is the outgoing variance and w = u - gamma S, then:
        w^2 - (A + eps S) w - A gamma S = 0.

    The positive root gives u.
    """
    a = np.exp(-k * tau)
    S = exact_ou_transition_variance(k, D, tau)
    A = a * a * var_in

    w = 0.5 * (
        (A + eps * S)
        + np.sqrt((A + eps * S) ** 2 + 4.0 * A * gamma * S)
    )
    return w + gamma * S

def exact_variance_step(var_in, k, D, tau):
    a = np.exp(-k * tau)
    S = exact_ou_transition_variance(k, D, tau)
    return a * a * var_in + S

def run_variance_test(eps, gamma):
    var_num = var0
    var_exact = var0
    max_err = 0.0

    nsteps = int(round(T / tau))
    for _ in range(nsteps):
        var_num = regularized_gaussian_variance_step(
            var_num, k, D, tau, eps, gamma
        )
        var_exact = exact_variance_step(var_exact, k, D, tau)
        max_err = max(max_err, abs(var_num - var_exact))

    return var_num, var_exact, max_err

def cyclic_shift(t0, tau, k, F0, omega):
    """
    Exact deterministic forcing shift over [t0,t0+tau] for
    F(t)=F0 cos(omega t):
        g_k = integral_0^tau exp[-k(tau-s)] F(t0+s) ds.
    """
    t1 = t0 + tau
    a = np.exp(-k * tau)

    return F0 / (k * k + omega * omega) * (
        k * np.cos(omega * t1)
        + omega * np.sin(omega * t1)
        - a * (
            k * np.cos(omega * t0)
            + omega * np.sin(omega * t0)
        )
    )

def gaussian_masses(x, mean, var):
    dx = x[1] - x[0]
    rho = np.exp(-0.5 * (x - mean) ** 2 / var)
    rho /= np.sqrt(2.0 * np.pi * var)
    p = rho * dx
    return p / p.sum()

def mass_moments(p, x):
    mean = np.sum(p * x)
    var = np.sum(p * (x - mean) ** 2)
    return mean, var

def markov_kernel_step(p, x, t0):
    """
    eps=1, gamma=0 endpoint.

    Since c_tau(x,y) = -log p_tau(y|x) + constant,
    exp(-c_tau) is the OU transition density up to normalization.

    Therefore the update is simply row-normalized Markov propagation.
    """
    a = np.exp(-k * tau)
    S = exact_ou_transition_variance(k, D, tau)
    g = cyclic_shift(t0, tau, k, F0, omega)

    mu = a * x[:, None] + g
    C = 0.5 * (x[None, :] - mu) ** 2 / S
    K = np.exp(-C)

    # Row-normalized transition probabilities on the finite grid.
    Tmat = K / K.sum(axis=1, keepdims=True)
    return p @ Tmat

print("\nCORRECTED FAMILY: gamma=(1-eps)/2")
print("eps      gamma      final variance    exact variance    final error      max error")
for eps in [0.00, 0.05, 0.10, 0.20, 0.50, 1.00]:
    gamma = 0.5 * (1.0 - eps)
    vn, ve, me = run_variance_test(eps, gamma)
    print(
        f"{eps:4.2f}    {gamma:7.4f}      "
        f"{vn:12.9f}    {ve:12.9f}    "
        f"{vn-ve:+.3e}    {me:.3e}"
    )

print("\nOLD CHOICE: gamma=1/2")
print("eps      eps+2gamma    final variance    exact variance    final error")
for eps in [0.05, 0.10, 0.20, 0.50, 1.00]:
    gamma = 0.5
    vn, ve, _ = run_variance_test(eps, gamma)
    print(
        f"{eps:4.2f}       {eps + 2*gamma:7.4f}       "
        f"{vn:12.9f}    {ve:12.9f}    {vn-ve:+.3e}"
    )

# ------------------------------------------------------------
# Direct Markov-kernel endpoint test: eps=1, gamma=0
# ------------------------------------------------------------
x = np.linspace(-6.0, 6.0, 641)
p = gaussian_masses(x, mean0, var0)

mean_exact = mean0
var_exact = var0
max_mean_err = 0.0
max_var_err = 0.0

nsteps = int(round(T / tau))
for n in range(nsteps):
    t0 = n * tau

    a = np.exp(-k * tau)
    S = exact_ou_transition_variance(k, D, tau)
    g = cyclic_shift(t0, tau, k, F0, omega)

    mean_exact = a * mean_exact + g
    var_exact = a * a * var_exact + S

    p = markov_kernel_step(p, x, t0)
    mean_num, var_num = mass_moments(p, x)

    max_mean_err = max(max_mean_err, abs(mean_num - mean_exact))
    max_var_err = max(max_var_err, abs(var_num - var_exact))

print("\nEPS=1, GAMMA=0: DIRECT OU MARKOV-KERNEL PROPAGATION")
print(f"max mean error over [0,{T:g}] = {max_mean_err:.3e}")
print(f"max variance error over [0,{T:g}] = {max_var_err:.3e}")
print("(Remaining error is only finite-grid / domain-truncation error.)")
