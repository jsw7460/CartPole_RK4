from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


# Type alias for ODE right-hand side function: f(t, y) -> dy/dt
ODEFunction = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class IntegratorResult:
    """Container for integration results."""

    t: NDArray[np.float64]  # Time points
    y: NDArray[np.float64]  # State history, shape: (n_steps + 1, state_dim)


class Integrator(ABC):
    """Abstract base class for ODE integrators."""

    def __init__(self, f: ODEFunction, dt: float):
        """
        Args:
            f: ODE right-hand side function dy/dt = f(t, y)
            dt: Time step size
        """
        self.f = f
        self.dt = dt

    @abstractmethod
    def step(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform a single integration step.

        Args:
            t: Current time
            y: Current state

        Returns:
            State at t + dt
        """
        pass

    def integrate(
        self, y0: NDArray[np.float64], t_span: tuple[float, float]
    ) -> IntegratorResult:
        """
        Integrate ODE over time span.

        Args:
            y0: Initial state
            t_span: (t_start, t_end)

        Returns:
            IntegratorResult containing time points and state history
        """
        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / self.dt))

        t = np.linspace(t_start, t_start + n_steps * self.dt, n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        y[0] = y0

        for i in range(n_steps):
            y[i + 1] = self.step(t[i], y[i])

        return IntegratorResult(t=t, y=y)


class ForwardEuler(Integrator):
    """Explicit Euler method (1st order)."""

    def step(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return y + self.dt * self.f(t, y)


class RK4(Integrator):
    """Classical 4th-order Runge-Kutta method."""

    def step(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        dt = self.dt

        k1 = self.f(t, y)
        k2 = self.f(t + dt / 2, y + dt * k1 / 2)
        k3 = self.f(t + dt / 2, y + dt * k2 / 2)
        k4 = self.f(t + dt, y + dt * k3)

        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class ImplicitEuler(Integrator):
    """
    Implicit Euler method (1st order, A-stable).

    Uses Newton's method for solving the implicit equation.
    """

    def __init__(
        self,
        f: ODEFunction,
        dt: float,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
        """
        Args:
            f: ODE right-hand side function
            dt: Time step size
            tol: Newton iteration tolerance
            max_iter: Maximum Newton iterations
        """
        super().__init__(f, dt)
        self.tol = tol
        self.max_iter = max_iter

    def step(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        dt = self.dt
        t_next = t + dt

        # Initial guess: Forward Euler
        y_next = y + dt * self.f(t, y)

        # Newton iteration to solve: y_next = y + dt * f(t_next, y_next)
        for _ in range(self.max_iter):
            residual = y_next - y - dt * self.f(t_next, y_next)

            if np.linalg.norm(residual) < self.tol:
                break

            # Approximate Jacobian via finite differences
            jac = self._approximate_jacobian(t_next, y_next)
            delta = np.linalg.solve(jac, -residual)
            y_next = y_next + delta

        return y_next

    def _approximate_jacobian(
        self, t: float, y: NDArray[np.float64], eps: float = 1e-8
    ) -> NDArray[np.float64]:
        """Approximate Jacobian of residual using finite differences."""
        n = len(y)
        jac = np.eye(n)

        for j in range(n):
            y_plus = y.copy()
            y_plus[j] += eps
            df = (self.f(t, y_plus) - self.f(t, y)) / eps
            jac[:, j] -= self.dt * df

        return jac