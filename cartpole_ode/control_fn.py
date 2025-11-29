from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Control input function type: F(t, state) -> force
ControlFunction = Callable[[float, NDArray[np.float64]], float]


def constant_force(force: float) -> ControlFunction:
    """Constant force applied to cart."""

    def control(t: float, state: NDArray[np.float64]) -> float:
        return force

    return control


def sinusoidal_force(amplitude: float, frequency: float) -> ControlFunction:
    """Sinusoidal forcing."""

    def control(t: float, state: NDArray[np.float64]) -> float:
        return amplitude * np.sin(2 * np.pi * frequency * t)

    return control


def bang_bang_control(force: float, target_theta: float = 0.0) -> ControlFunction:
    """Simple bang-bang controller to balance pole."""

    def control(t: float, state: NDArray[np.float64]) -> float:
        theta = state[2]
        theta_dot = state[3]

        # Switch based on angle and angular velocity
        if theta > target_theta or (theta == target_theta and theta_dot > 0):
            return -force
        else:
            return force

    return control


def pd_control(kp: float, kd: float, target_theta: float = 0.0) -> ControlFunction:
    """PD controller for pole balancing."""

    def control(t: float, state: NDArray[np.float64]) -> float:
        theta = state[2]
        theta_dot = state[3]

        error = theta - target_theta
        return -kp * error - kd * theta_dot

    return control


def lqr_control(K: NDArray[np.float64]) -> ControlFunction:
    """
    LQR controller with precomputed gain matrix.

    Args:
        K: Gain matrix [1 x 4], control u = -K @ state
    """

    def control(t: float, state: NDArray[np.float64]) -> float:
        return -K @ state

    return control


def impulse_force(force: float, t_start: float, duration: float) -> ControlFunction:
    """Single impulse at specified time."""

    def control(t: float, state: NDArray[np.float64]) -> float:
        if t_start <= t < t_start + duration:
            return force
        return 0.0

    return control


def swing_up_energy_control(
    params: "CartPoleParams",
    k: float = 1.0,
    target_energy: float | None = None
) -> ControlFunction:
    """
    Energy-based swing-up controller.

    Pumps energy into system until reaching upright position energy.
    """
    if target_energy is None:
        # Energy at upright position (theta=0)
        target_energy = params.m_p * params.g * params.L

    def control(t: float, state: NDArray[np.float64]) -> float:
        theta = state[2]
        theta_dot = state[3]

        # Current energy (relative to hanging position)
        E = 0.5 * params.m_p * (params.L * theta_dot) ** 2 \
            + params.m_p * params.g * params.L * (1 + np.cos(theta))

        # Pump energy in direction of motion
        energy_error = E - target_energy
        return -k * energy_error * np.sign(theta_dot * np.cos(theta))

    return control