from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class CartPoleParams:
    """Physical parameters for Cart-Pole system."""

    M: float = 1.0  # Cart mass [kg]
    m_p: float = 1.0  # Pole point mass [kg]
    L: float = 1.0  # Pole length [m]
    g: float = 9.81  # Gravitational acceleration [m/s^2]


# Control input function type: F(t, state) -> force
ControlFunction = Callable[[float, NDArray[np.float64]], float]


def no_control(t: float, state: NDArray[np.float64]) -> float:
    """Zero control input (free dynamics)."""
    return 0.0


class CartPoleDynamics:
    """
    Cart-Pole equations of motion.

    State vector: [x, x_dot, theta, theta_dot]
        x: Cart position
        x_dot: Cart velocity
        theta: Pole angle (counterclockwise from vertical y-axis)
        theta_dot: Pole angular velocity

    Equations derived from Newton-Euler formulation:
        F = (M + m_p) * x_ddot + m_p * L * theta_dot^2 * sin(theta)
            - m_p * L * theta_ddot * cos(theta)
        m_p * L^2 * theta_ddot - m_p * L * g * sin(theta)
            - m_p * L * x_ddot * cos(theta) = 0
    """

    # State indices
    X = 0
    X_DOT = 1
    THETA = 2
    THETA_DOT = 3

    def __init__(
        self,
        params: CartPoleParams | None = None,
        control: ControlFunction = no_control,
    ):
        """
        Args:
            params: Physical parameters (uses defaults if None)
            control: Control function F(t, state) -> force on cart
        """
        self.params = params or CartPoleParams()
        self.control = control

    def __call__(
        self, t: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute state derivative.

        Args:
            t: Current time
            state: Current state [x, x_dot, theta, theta_dot]

        Returns:
            State derivative [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        p = self.params
        x_dot = state[self.X_DOT]
        theta = state[self.THETA]
        theta_dot = state[self.THETA_DOT]

        F = self.control(t, state)

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        denom = p.M + p.m_p * sin_th**2

        x_ddot = (
            F
            - p.m_p * p.L * theta_dot**2 * sin_th
            + p.m_p * p.g * sin_th * cos_th
        ) / denom

        theta_ddot = (
            F * cos_th
            + (p.M + p.m_p) * p.g * sin_th
            - p.m_p * p.L * theta_dot**2 * sin_th * cos_th
        ) / (p.L * denom)

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def get_pole_position(
        self, state: NDArray[np.float64]
    ) -> tuple[float, float]:
        """
        Compute pole tip position.

        Args:
            state: State vector

        Returns:
            (x_pole, y_pole) coordinates of pole tip
        """
        x = state[self.X]
        theta = state[self.THETA]

        x_pole = x + self.params.L * np.sin(theta)
        y_pole = self.params.L * np.cos(theta)

        return x_pole, y_pole