import matplotlib.pyplot as plt
import numpy as np

from cartpole_ode.dynamics import CartPoleDynamics, CartPoleParams
from cartpole_ode.integrators import RK4, ForwardEuler
from cartpole_ode import control_fn
from cartpole_ode.visualization import (
    plot_state_history,
    plot_phase_portrait,
    plot_energy,
)


def main():
    # =========================================================================
    # Configuration
    # =========================================================================

    params = CartPoleParams(
        M=1.0,  # Cart mass [kg]
        m_p=1.5,  # Pole point mass [kg]
        L=2.0,  # Pole length [m]
        g=9.81,  # Gravity [m/s^2]
    )

    init_degree = 30.0
    # Initial conditions: [x, x_dot, theta, theta_dot]
    y0 = np.array([
        0.0,  # x: cart at origin
        0.0,  # x_dot: cart at rest
        init_degree * np.pi / 180,  # theta:
        0.0,  # theta_dot: no initial angular velocity
    ])

    t_span = (0.0, 6.0)
    dt = 0.005

    # =========================================================================
    # Simulation
    # =========================================================================

    dynamics = CartPoleDynamics(
        params=params,
        # control=control_fn.pd_control(20.0, 10.0)
    )

    # Choose integrator
    integrator = RK4(f=dynamics, dt=dt)
    # integrator = ForwardEuler(f=dynamics, dt=dt)
    # integrator = ImplicitEuler(f=dynamics, dt=dt)

    result = integrator.integrate(y0=y0, t_span=t_span)

    # =========================================================================
    # Visualization
    # =========================================================================
    from cartpole_ode.visualization import create_animation

    anim = create_animation(result, params, skip_frames=10)
    anim.save("cartpole.gif", writer="pillow", fps=30)
    plt.close()

    # 나머지 플롯 표시
    fig1 = plot_state_history(result, title=f"Cart-Pole Simulation (RK4, dt={dt})")
    fig2 = plot_phase_portrait(result)
    fig3 = plot_energy(result, params)
    plt.show()



if __name__ == "__main__":
    main()
