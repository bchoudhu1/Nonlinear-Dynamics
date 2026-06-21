import numpy as np
import matplotlib.pyplot as plt

POWER_PERIODS = [1, 2, 4, 8, 16, 32, 64, 128]


# -----------------------------
# Map definitions
# -----------------------------
def logistic_step(x, r):
    return r * x * (1 - x)


def quadratic_step(x, c):
    return x * x + c


def iterate_map(step_func, param, x0, steps, escape_radius=None):
    """
    Iterate x_{n+1} = f(x_n, param).

    Returns:
        np.ndarray of length steps + 1, or None if the orbit escapes.
    """
    x = np.empty(steps + 1, dtype=float)
    x[0] = x0

    for i in range(steps):
        xn = step_func(x[i], param)

        if not np.isfinite(xn):
            return None

        if escape_radius is not None and abs(xn) > escape_radius:
            return None

        x[i + 1] = xn

    return x


# -----------------------------
# Period detection
# -----------------------------
def orbit_period(steady, tol=1e-8):
    """
    Estimate the period of the steady-state tail.
    Only checks powers of 2, which is what we want for the main
    period-doubling cascade.
    """
    if steady is None:
        return None

    x = np.asarray(steady, dtype=float)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        return None

    n = len(x)

    for p in POWER_PERIODS:
        if 2 * p > n:
            break

        a = x[-2 * p : -p]
        b = x[-p :]

        if np.max(np.abs(a - b)) < tol:
            return p

    return None


def period_at(step_func, param, x0, steps, discard, escape_radius=None):
    traj = iterate_map(step_func, param, x0, steps, escape_radius=escape_radius)
    if traj is None:
        return None
    return orbit_period(traj[discard:])


# -----------------------------
# Bifurcation detection
# -----------------------------
def find_period_doubling_brackets(step_func, params, x0, steps, discard, escape_radius=None):
    """
    Scan once and find coarse brackets [left, right] where
    the orbit first doubles from p to 2p.

    Returns a list of tuples:
        (left_param, right_param, target_period)
    """
    brackets = []
    expected = 1
    last_good_param = None

    for param in params:
        p = period_at(step_func, param, x0, steps, discard, escape_radius=escape_radius)

        if p == expected:
            last_good_param = param

        elif p == 2 * expected:
            left = last_good_param if last_good_param is not None else param
            right = param
            brackets.append((left, right, 2 * expected))
            expected *= 2
            last_good_param = param

            if expected >= 32:
                break

    return brackets


def refine_bifurcation(step_func, left, right, target_period, x0, steps, discard,
                       escape_radius=None, iterations=35):
    """
    Binary search the boundary where the detected period reaches target_period.
    """
    if left > right:
        left, right = right, left

    for _ in range(iterations):
        mid = 0.5 * (left + right)
        p = period_at(step_func, mid, x0, steps, discard, escape_radius=escape_radius)

        if p is None:
            right = mid
        elif p < target_period:
            left = mid
        else:
            right = mid

    return 0.5 * (left + right)


def detect_and_refine(step_func, params, x0, steps, discard, escape_radius=None):
    coarse = find_period_doubling_brackets(
        step_func, params, x0, steps, discard, escape_radius=escape_radius
    )

    refined = []
    for left, right, target_period in coarse:
        left, right = sorted([left, right])
        param = refine_bifurcation(
            step_func, left, right, target_period, x0, steps, discard,
            escape_radius=escape_radius
        )
        refined.append((param, target_period))

    return refined


def feigenbaum_estimates(bifurcations):
    """
    bifurcations = [(param_1, period_1), (param_2, period_2), ...]
    Returns successive Feigenbaum delta estimates.
    """
    params = [param for param, _ in bifurcations]
    if len(params) < 3:
        return []

    ests = []
    for i in range(1, len(params) - 1):
        delta = (params[i] - params[i - 1]) / (params[i + 1] - params[i])
        ests.append(delta)

    return ests


# -----------------------------
# Plot helper
# -----------------------------
def plot_bifurcation(ax, step_func, params, x0, steps, discard, title,
                     xlabel, escape_radius=None, y_lim=None):
    for param in params:
        traj = iterate_map(step_func, param, x0, steps, escape_radius=escape_radius)
        if traj is None:
            continue

        steady = traj[discard:]
        ax.scatter(
            np.full(len(steady), param),
            steady,
            s=0.1,
            c='black',
            alpha=0.8
        )

    bifurcations = detect_and_refine(
        step_func, params, x0, steps, discard, escape_radius=escape_radius
    )

    for param, period in bifurcations:
        ax.axvline(param, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(
            param,
            1.02 if y_lim is not None and y_lim[1] <= 1.2 else 1.05,
            f"{period}",
            rotation=90,
            va='bottom',
            ha='center',
            fontsize=8,
            color='red'
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("x")

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    return bifurcations


# -----------------------------
# Parameters
# -----------------------------
log_steps = 10000
log_discard = 9000

quad_steps = 30000
quad_discard = 29000

# Logistic map: x_{n+1} = r x_n (1 - x_n)
x0_logistic = 0.2
r_min, r_max, r_step = 2.4, 4.0, 0.001
r_vals = np.arange(r_min, r_max, r_step)

# Quadratic family: x_{n+1} = x_n^2 + c
# Scan descending, because the main cascade runs toward more negative c.
x0_quadratic = 0.0
c_min, c_max, c_step = -1.45, -0.70, 0.001
c_vals = np.arange(c_max, c_min, -c_step)


# -----------------------------
# Plot side by side
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

log_bifs = plot_bifurcation(
    axes[0],
    logistic_step,
    r_vals,
    x0_logistic,
    log_steps,
    log_discard,
    title="Logistic map bifurcation diagram",
    xlabel="r",
    escape_radius=None,
    y_lim=(0, 1.05)
)

quad_bifs = plot_bifurcation(
    axes[1],
    quadratic_step,
    c_vals,
    x0_quadratic,
    quad_steps,
    quad_discard,
    title="Quadratic map bifurcation diagram",
    xlabel="c",
    escape_radius=2.0,
    y_lim=(-2.2, 2.2)
)

plt.tight_layout()
plt.show()


# -----------------------------
# Feigenbaum estimates
# -----------------------------
print("Logistic map bifurcations:")
for param, period in log_bifs:
    print(f"r ≈ {param:.6f} -> period {period}")

log_ests = feigenbaum_estimates(log_bifs)
if log_ests:
    print("Logistic Feigenbaum estimates:", log_ests)
    print("Latest logistic estimate:", log_ests[-1])
else:
    print("Not enough logistic bifurcation points for Feigenbaum estimate.")

print("\nQuadratic map bifurcations:")
for param, period in quad_bifs:
    print(f"c ≈ {param:.6f} -> period {period}")

quad_ests = feigenbaum_estimates(quad_bifs)
if quad_ests:
    print("Quadratic Feigenbaum estimates:", quad_ests)
    print("Latest quadratic estimate:", quad_ests[-1])
else:
    print("Not enough quadratic bifurcation points for Feigenbaum estimate.")