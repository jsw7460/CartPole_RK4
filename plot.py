import numpy as np
import matplotlib.pyplot as plt

from cartpole_ode.dynamics import CartPoleDynamics, CartPoleParams
from cartpole_ode.integrators import RK4

plt.rcParams.update({'font.size': 14})

def main():
    params = CartPoleParams(
        M=1.0,
        m_p=1.0,
        L=1.0,
        g=9.81,
    )

    init_degree = 30.0
    y0 = np.array([0.0, 0.0, init_degree * np.pi / 180, 0.0])
    t_span = (0.0, 10.0)

    dynamics = CartPoleDynamics(params=params)

    # Different step sizes
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]

    results = {}

    for h in step_sizes:
        integrator = RK4(f=dynamics, dt=h)
        results[h] = integrator.integrate(y0=y0, t_span=t_span)
        print(f"h = {h}: {len(results[h].t)} steps")

    # Reference solution
    h_ref = 0.0001
    integrator_ref = RK4(f=dynamics, dt=h_ref)
    result_ref = integrator_ref.integrate(y0=y0, t_span=t_span)
    print(f"h = {h_ref} (reference): {len(result_ref.t)} steps")

    # =========================================================================
    # Plot 1: Theta trajectories for different step sizes
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for h in step_sizes:
        ax1.plot(results[h].t, results[h].y[:, 2], label=f"h = {h}")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"$\theta$ [rad]")
    ax1.set_title("Convergence of RK4: Pole Angle")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.savefig("convergence_theta.png", dpi=150, bbox_inches="tight")

    # =========================================================================
    # Plot 2: Error vs step size (log-log)
    # =========================================================================
    errors = []
    for h in step_sizes:
        # Interpolate to compare at same time points
        t_coarse = results[h].t
        theta_coarse = results[h].y[:, 2]

        # Find corresponding indices in reference
        indices = (t_coarse / h_ref).astype(int)
        indices = np.clip(indices, 0, len(result_ref.t) - 1)
        theta_ref_interp = result_ref.y[indices, 2]

        max_error = np.max(np.abs(theta_coarse - theta_ref_interp))
        errors.append(max_error)
        print(f"h = {h}: max error = {max_error:.6e}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.loglog(step_sizes, errors, "o-", linewidth=2, markersize=8)

    # Reference slope for O(h^4)
    h_line = np.array([step_sizes[0], step_sizes[-1]])
    e_line = errors[0] * (h_line / step_sizes[0]) ** 4
    ax2.loglog(h_line, e_line, "--", color="gray", label=r"$O(h^4)$ reference")

    ax2.set_xlabel("Step size h")
    ax2.set_ylabel("Maximum error")
    ax2.set_title("RK4 Convergence Order")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")
    fig2.savefig("convergence_order.png", dpi=150, bbox_inches="tight")

    # =========================================================================
    # Plot 3: Energy conservation for different step sizes
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for h in step_sizes:
        res = results[h]
        x_dot = res.y[:, 1]
        theta = res.y[:, 2]
        theta_dot = res.y[:, 3]

        T = 0.5 * (params.M + params.m_p) * x_dot ** 2 \
            - params.m_p * params.L * x_dot * theta_dot * np.cos(theta) \
            + 0.5 * params.m_p * params.L ** 2 * theta_dot ** 2
        V = params.m_p * params.g * params.L * np.cos(theta)
        E = T + V

        # Normalize by initial energy
        E_error = (E - E[0]) / np.abs(E[0])
        ax3.plot(res.t, E_error, label=f"h = {h}")

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Relative Energy Error $(E - E_0) / |E_0|$")
    ax3.set_title("Energy Conservation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.savefig("energy_conservation.png", dpi=150, bbox_inches="tight")

    # =========================================================================
    # Print table
    # =========================================================================
    print("\n| Step size h | Max Error      | Ratio |")
    print("|-------------|----------------|-------|")
    for i, h in enumerate(step_sizes):
        if i == 0:
            print(f"| {h:<11} | {errors[i]:<14.6e} | -     |")
        else:
            ratio = errors[i - 1] / errors[i]
            print(f"| {h:<11} | {errors[i]:<14.6e} | {ratio:.1f}  |")

    plt.show()


if __name__ == "__main__":
    main()