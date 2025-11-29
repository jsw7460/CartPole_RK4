import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from .dynamics import CartPoleParams
from .integrators import IntegratorResult


def plot_state_history(
    result: IntegratorResult,
    title: str = "Cart-Pole State History",
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Plot state variables over time.

    Args:
        result: Integration result
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    labels = [
        (r"$x$ [m]", "Cart Position"),
        (r"$\dot{x}$ [m/s]", "Cart Velocity"),
        (r"$\theta$ [rad]", "Pole Angle"),
        (r"$\dot{\theta}$ [rad/s]", "Pole Angular Velocity"),
    ]

    for i, ax in enumerate(axes.flat):
        ax.plot(result.t, result.y[:, i], linewidth=1.5)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(labels[i][0])
        ax.set_title(labels[i][1])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_phase_portrait(
    result: IntegratorResult,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Plot phase portraits for cart and pole.

    Args:
        result: Integration result
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Phase Portraits", fontsize=14)

    # Cart phase portrait
    axes[0].plot(result.y[:, 0], result.y[:, 1], linewidth=1.5)
    axes[0].scatter(result.y[0, 0], result.y[0, 1], c="green", s=50,
                    label="Start", zorder=5)
    axes[0].scatter(result.y[-1, 0], result.y[-1, 1], c="red", s=50,
                    label="End", zorder=5)
    axes[0].set_xlabel(r"$x$ [m]")
    axes[0].set_ylabel(r"$\dot{x}$ [m/s]")
    axes[0].set_title("Cart")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Pole phase portrait
    axes[1].plot(result.y[:, 2], result.y[:, 3], linewidth=1.5)
    axes[1].scatter(result.y[0, 2], result.y[0, 3], c="green", s=50,
                    label="Start", zorder=5)
    axes[1].scatter(result.y[-1, 2], result.y[-1, 3], c="red", s=50,
                    label="End", zorder=5)
    axes[1].set_xlabel(r"$\theta$ [rad]")
    axes[1].set_ylabel(r"$\dot{\theta}$ [rad/s]")
    axes[1].set_title("Pole")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_energy(
    result: IntegratorResult,
    params: CartPoleParams,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot kinetic, potential, and total energy over time.

    Args:
        result: Integration result
        params: Cart-Pole parameters
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    x_dot = result.y[:, 1]
    theta = result.y[:, 2]
    theta_dot = result.y[:, 3]

    # Kinetic energy
    T_cart = 0.5 * params.M * x_dot ** 2
    T_pole = 0.5 * params.m_p * (
        x_dot ** 2
        - 2 * x_dot * params.L * theta_dot * np.cos(theta)
        + (params.L * theta_dot) ** 2
    )
    T = T_cart + T_pole

    # Potential energy (pole tip height)
    V = params.m_p * params.g * params.L * np.cos(theta)

    # Total energy
    E = T + V

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result.t, T, label="Kinetic Energy", linewidth=1.5)
    ax.plot(result.t, V, label="Potential Energy", linewidth=1.5)
    ax.plot(result.t, E, label="Total Energy", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.set_title("Energy Conservation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_animation(
    result: IntegratorResult,
    params: CartPoleParams,
    skip_frames: int = 1,
    figsize: tuple[float, float] = (10, 6),
) -> FuncAnimation:
    """
    Create animation of Cart-Pole motion.

    Args:
        result: Integration result
        params: Cart-Pole parameters
        skip_frames: Plot every n-th frame (for faster rendering)
        figsize: Figure size

    Returns:
        Matplotlib FuncAnimation object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine axis limits
    x_data = result.y[:, 0]
    x_margin = params.L + 0.5
    x_min, x_max = x_data.min() - x_margin, x_data.max() + x_margin
    y_min, y_max = -0.5, params.L + 0.5

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Cart dimensions
    cart_width, cart_height = 0.4, 0.2

    # Initialize plot elements
    cart = Rectangle((0, 0), cart_width, cart_height, fc="blue", ec="black")
    ax.add_patch(cart)
    (pole_line,) = ax.plot([], [], "o-", color="orange", linewidth=3,
                           markersize=10)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        cart.set_xy((-cart_width / 2, -cart_height / 2))
        pole_line.set_data([], [])
        time_text.set_text("")
        return cart, pole_line, time_text

    def animate(frame_idx):
        idx = frame_idx * skip_frames
        x = result.y[idx, 0]
        theta = result.y[idx, 2]

        # Update cart
        cart.set_xy((x - cart_width / 2, -cart_height / 2))

        # Update pole
        pole_x = [x, x - params.L * np.sin(theta)]
        pole_y = [0, params.L * np.cos(theta)]
        pole_line.set_data(pole_x, pole_y)

        # Update time
        time_text.set_text(f"t = {result.t[idx]:.2f} s")

        return cart, pole_line, time_text

    n_frames = len(result.t) // skip_frames
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=20, blit=True
    )

    return anim
