import matplotlib.pyplot as plt
# -----------------------------
# One-step methods
# -----------------------------

def forward_euler_onestep(init_conds, dt, k, m):
    x_primed = [init_conds[1], -(k / m) * init_conds[0]]
    update_var = [dt * i for i in x_primed]
    x_new = [x_i + upd_var for x_i, upd_var in zip(init_conds, update_var)]
    return x_new


def backward_euler_onestep(init_conds, dt, k, m):
    # Predictor-corrector style approximation
    forward_euler_res = forward_euler_onestep(init_conds, dt, k, m)
    x_primed = [forward_euler_res[1], -(k / m) * forward_euler_res[0]]
    update_var = [dt * i for i in x_primed]
    x_new = [x_i + upd_var for x_i, upd_var in zip(init_conds, update_var)]
    return x_new


def trapezoidal_onestep(init_conds, dt, k, m):
    # Predictor-corrector style approximation
    x_primed_fwd = [init_conds[1], -(k / m) * init_conds[0]]
    update_var_fwd = [dt * i for i in x_primed_fwd]
    x_new_fwd = [x_i + upd_var for x_i, upd_var in zip(init_conds, update_var_fwd)]

    x_primed_back = [x_new_fwd[1], -(k / m) * x_new_fwd[0]]
    update_var_back = [dt * i for i in x_primed_back]

    update_var_mean = [
        (upd_1 + upd_2) / 2
        for upd_1, upd_2 in zip(update_var_fwd, update_var_back)
    ]
    x_new_trap = [x_i + upd_var for x_i, upd_var in zip(init_conds, update_var_mean)]
    return x_new_trap


def rk4_onestep(init_conds, h, rhs_funct, *args):
    k1 = rhs_funct(init_conds, *args)
    k2 = rhs_funct(
        [x + (h / 2) * k1_i for x, k1_i in zip(init_conds, k1)],
        *args
    )
    k3 = rhs_funct(
        [x + (h / 2) * k2_i for x, k2_i in zip(init_conds, k2)],
        *args
    )
    k4 = rhs_funct(
        [x + h * k3_i for x, k3_i in zip(init_conds, k3)],
        *args
    )

    x_new = [
        x + (h / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        for x, k1_i, k2_i, k3_i, k4_i in zip(init_conds, k1, k2, k3, k4)
    ]
    return x_new


def adaptive_rk4_onestep(state, dt, tol, rhs_funct, *args):
    h = dt

    while True:
        # One full step
        x_full = rk4_onestep(state, h, rhs_funct, *args)

        # Two half steps
        x_half = rk4_onestep(state, h / 2, rhs_funct, *args)
        x_two_half = rk4_onestep(x_half, h / 2, rhs_funct, *args)

        err = max(abs(a - b) for a, b in zip(x_full, x_two_half))

        if err > tol:
            h = h / 2
            continue

        if err < tol / 16:
            h_next = 2 * h
        else:
            h_next = h

        return {
            "state": x_two_half,
            "accepted_step": h,
            "next_step": h_next,
            "error": err
        }
# -----------------------------
# Example RHS: Lorenz system
# -----------------------------

def lorenz_rhs(state, sigma, rho, beta):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]


# -----------------------------
# Generic trajectory generators
# -----------------------------

def generate_traj(init_conds, step, tot_time, method, *args):
    total_time = 0.0
    iters = int(tot_time / step)
    x_old = init_conds
    x_new = init_conds

    for _ in range(iters):
        x_new = method(x_old, step, *args)
        x_old = x_new
        total_time += step

    return x_new, total_time


def generate_trajectory(steps, init_conds, step, method, *args):
    x_vals = [init_conds]
    times = [0.0]

    x_old = init_conds
    total_time = 0.0

    for _ in range(steps):
        x_new = method(x_old, step, *args)
        x_vals.append(x_new)
        total_time += step
        times.append(total_time)
        x_old = x_new

    return x_vals, times


def plot_state_space(steps, init_conds, step, k, m, method, label=None):
    x_vals, _ = generate_trajectory(steps, init_conds, step, method, k, m)

    x_pos = [state[0] for state in x_vals]
    x_vel = [state[1] for state in x_vals]

    plt.plot(x_pos, x_vel, label=label)

def generate_adaptive_trajectory(init_conds, dt, tot_time, tol, method, *args):
    t = 0.0
    state = init_conds[:]

    times = [t]
    states = [state]
    h_values = []

    while t < tot_time:
        dt_use = min(dt, tot_time - t)
        step_info = method(state, dt_use, tol, *args)

        state = step_info["state"]
        accepted_step = step_info["accepted_step"]
        dt = step_info["next_step"]

        h_values.append(accepted_step)
        t += accepted_step

        times.append(t)
        states.append(state)

    return states, times, h_values

# -----------------------------
# Example usage
# -----------------------------
x_init = [-1, -2]
k = 2
m = 0.5
step = 0.05
tot_time = 0.5

x_new, total_time = generate_traj(
    x_init, step, tot_time, trapezoidal_onestep, k, m
)

print("The final point of the trajectory is at:", x_new)
print("Total time for the trajectory is:", total_time)


# 500-point trajectory
plt.figure()
plot_state_space(
    500, [-1, -2], 0.01, 2, 0.5,
    trapezoidal_onestep,
    "Trapezoidal"
)
plt.xlabel("Position x")
plt.ylabel("Velocity v")
plt.title("State Space Plot, 500 points")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


# 5000-point trajectory
plt.figure()
plot_state_space(
    5000, [-1, -2], 0.01, 2, 0.5,
    trapezoidal_onestep,
    "Trapezoidal"
)
plt.xlabel("Position x")
plt.ylabel("Velocity v")
plt.title("State Space Plot, 5000 points")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


#Overlay both on a single plot 
plt.figure(figsize=(8, 8))

plot_state_space(
    500, [-1, -2], 0.01, 2, 0.5,
    trapezoidal_onestep,
    "500 points"
)

plot_state_space(
    5000, [-1, -2], 0.01, 2, 0.5,
    trapezoidal_onestep,
    "5000 points"
)

plt.xlabel("Position x")
plt.ylabel("Velocity v")
plt.title("State Space Comparison")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Generate Lorenz trajectory - Adaptive RK4
# -----------------------------

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

x_init = [1.0, 1.0, 1.0]
dt = 0.01
tot_time = 40.0
tol = 1e-6

states, times, h_values = generate_adaptive_trajectory(
    x_init, dt, tot_time, tol,
    adaptive_rk4_onestep,
    lorenz_rhs, sigma, rho, beta
)

x_vals = [s[0] for s in states]
y_vals = [s[1] for s in states]
z_vals = [s[2] for s in states]

print("Final state:", states[-1])
print("Final time:", times[-1])
print("Final h:", h_values[-1])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, lw=0.8)

ax.set_title(f"Lorenz Trajectory (Adaptive RK4), final h = {h_values[-1]:.3e}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.text2D(
    0.05, 0.95,
    f"final h = {h_values[-1]:.3e}",
    transform=ax.transAxes
)

plt.show()