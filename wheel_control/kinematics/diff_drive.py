"""Differential drive kinematics model with first-order actuator dynamics."""

from __future__ import annotations

import numpy as np

from ..utils.math_utils import MathUtils


class DiffDriveKinematics:
    """Differential drive robot kinematic model.

    State: [x, y, theta, vx, omega]
    Non-holonomic constraint: vy = 0 (no lateral slip).
    Includes configurable first-order actuator inertia and acceleration limits.
    """

    def __init__(
        self,
        wheel_base: float = 0.3,
        wheel_radius: float = 0.05,
        max_v: float = 1.5,
        max_omega: float = 3.0,
        tau_v: float = 0.1,
        tau_omega: float = 0.08,
        v_acc_max: float = 3.0,
        w_acc_max: float = 10.0,
        dt: float = 0.02,
    ):
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.max_v = max_v
        self.max_omega = max_omega
        self.tau_v = tau_v
        self.tau_omega = tau_omega
        self.v_acc_max = v_acc_max
        self.w_acc_max = w_acc_max
        self.dt = dt

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.omega = 0.0

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        vx: float = 0.0,
        omega: float = 0.0,
    ) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = vx
        self.omega = omega

    @property
    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.vx, self.omega])

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return MathUtils.wrap_angle(angle)

    def step(self, v_cmd: float, omega_cmd: float) -> np.ndarray:
        """Advance one timestep given velocity commands.

        Returns the new state [x, y, theta, vx, omega].
        """
        v_cmd = float(np.clip(v_cmd, -self.max_v, self.max_v))
        omega_cmd = float(np.clip(omega_cmd, -self.max_omega, self.max_omega))

        # First-order actuator response
        v_err = v_cmd - self.vx
        w_err = omega_cmd - self.omega

        v_dot = v_err / self.tau_v
        w_dot = w_err / self.tau_omega

        # Acceleration clamping
        v_dot = float(np.clip(v_dot, -self.v_acc_max, self.v_acc_max))
        w_dot = float(np.clip(w_dot, -self.w_acc_max, self.w_acc_max))

        self.vx += v_dot * self.dt
        self.omega += w_dot * self.dt

        self.vx = float(np.clip(self.vx, -self.max_v, self.max_v))
        self.omega = float(np.clip(self.omega, -self.max_omega, self.max_omega))

        # Midpoint kinematic integration (reduces heading drift vs forward Euler)
        theta_mid = self.theta + self.omega * self.dt * 0.5
        self.x += self.vx * np.cos(theta_mid) * self.dt
        self.y += self.vx * np.sin(theta_mid) * self.dt
        self.theta = self.wrap_angle(self.theta + self.omega * self.dt)

        return self.state

    def cmd_to_wheel_speed(self, v: float, omega: float) -> tuple[float, float]:
        """Convert (v, omega) to individual wheel angular velocities (rad/s)."""
        v_left = (v - omega * self.wheel_base / 2.0) / self.wheel_radius
        v_right = (v + omega * self.wheel_base / 2.0) / self.wheel_radius
        return v_left, v_right
