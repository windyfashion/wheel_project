"""LQR Controller for differential drive trajectory tracking."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .base import ControllerBase, ControlOutput
from ..utils.frenet import FrenetFrame


class LQRController(ControllerBase):
    """Linear Quadratic Regulator controller for trajectory tracking.

    Uses Frenet frame error coordinates:
    - State: [e_lat, e_yaw, e_v, e_omega]
    - Control: [delta_v, delta_omega]

    The system is linearized around the reference trajectory at each step,
    including curvature coupling and first-order actuator dynamics.
    """

    def __init__(
        self,
        dt: float = 0.02,
        Q: np.ndarray | list | None = None,
        R: np.ndarray | list | None = None,
        tau_v: float = 0.1,
        tau_omega: float = 0.08,
        max_v: float = 1.5,
        max_omega: float = 3.0,
    ):
        """Initialize LQR controller.

        Parameters
        ----------
        dt : float
            Control timestep
        Q : ndarray or list, shape (4,) or (4, 4)
            State weight matrix diagonal [e_lat, e_yaw, e_v, e_omega]
        R : ndarray or list, shape (2,) or (2, 2)
            Control weight matrix diagonal [delta_v, delta_omega]
        tau_v : float
            Linear velocity actuator time constant
        tau_omega : float
            Angular velocity actuator time constant
        max_v : float
            Maximum linear velocity for output clipping
        max_omega : float
            Maximum angular velocity for output clipping
        """
        super().__init__(dt)

        if Q is None:
            Q = [1.0, 2.0, 0.5, 0.5]
        if R is None:
            R = [0.1, 0.1]

        self.Q = np.diag(Q) if np.ndim(Q) == 1 else np.asarray(Q, dtype=float)
        self.R = np.diag(R) if np.ndim(R) == 1 else np.asarray(R, dtype=float)
        self.tau_v = tau_v
        self.tau_omega = tau_omega
        self.max_v = max_v
        self.max_omega = max_omega

        self._K: np.ndarray | None = None
        self._last_kappa: float = 0.0
        self._last_v: float = 0.0

    def reset(self) -> None:
        """Reset controller internal state."""
        self._K = None
        self._last_kappa = 0.0
        self._last_v = 0.0

    def set_weights(self, Q: np.ndarray | list, R: np.ndarray | list) -> None:
        """Set new weight matrices.

        Parameters
        ----------
        Q : ndarray or list
            State weight (1-D diagonal or 2-D matrix)
        R : ndarray or list
            Control weight (1-D diagonal or 2-D matrix)
        """
        self.Q = np.diag(Q) if np.ndim(Q) == 1 else np.asarray(Q, dtype=float)
        self.R = np.diag(R) if np.ndim(R) == 1 else np.asarray(R, dtype=float)
        self._K = None

    def _build_system_matrices(
        self,
        ref_v: float,
        ref_omega: float,
        kappa: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build linearized and discretized system matrices A_d and B_d.

        Continuous-time Frenet error dynamics (linearized around e=0):

            e_lat_dot   =  ref_v * e_yaw
            e_yaw_dot   = -ref_v * kappa^2 * e_lat  -  kappa * e_v  +  e_omega
            e_v_dot     = -e_v / tau_v   +  delta_v / tau_v
            e_omega_dot = -e_omega / tau_w +  delta_omega / tau_w

        Discretization via matrix exponential on the augmented matrix
        [[A, B], [0, 0]] * dt  ->  expm gives [[A_d, B_d], [0, I]].

        Parameters
        ----------
        ref_v : float
            Reference velocity at current point
        ref_omega : float
            Reference angular velocity at current point
        kappa : float
            Curvature at current point

        Returns
        -------
        tuple[ndarray, ndarray]
            (A_d, B_d) discrete-time system matrices
        """
        tv = self.tau_v
        tw = self.tau_omega

        A = np.array([
            [0.0,               ref_v,  0.0,      0.0     ],
            [-ref_v * kappa**2, 0.0,    -kappa,   1.0     ],
            [0.0,               0.0,    -1.0/tv,  0.0     ],
            [0.0,               0.0,    0.0,      -1.0/tw ],
        ])

        B = np.array([
            [0.0,      0.0     ],
            [0.0,      0.0     ],
            [1.0/tv,   0.0     ],
            [0.0,      1.0/tw  ],
        ])

        n_x, n_u = 4, 2
        M = np.zeros((n_x + n_u, n_x + n_u))
        M[:n_x, :n_x] = A
        M[:n_x, n_x:] = B

        M_d = linalg.expm(M * self.dt)
        A_d = M_d[:n_x, :n_x]
        B_d = M_d[:n_x, n_x:]

        return A_d, B_d

    def _compute_gain_matrix(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Compute LQR gain matrix K using discrete-time Riccati equation.

        Parameters
        ----------
        A, B : ndarray
            Discrete-time system matrices

        Returns
        -------
        ndarray, shape (2, 4)
            LQR gain matrix K
        """
        try:
            P = linalg.solve_discrete_are(A, B, self.Q, self.R)
            K = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
            return K
        except (np.linalg.LinAlgError, ValueError):
            if self._K is not None:
                return self._K
            return np.zeros((2, 4))

    def compute_control(
        self,
        state: np.ndarray,
        ref_trajectory: np.ndarray,
        nearest_idx: int,
    ) -> ControlOutput:
        """Compute LQR control commands.

        Parameters
        ----------
        state : ndarray, shape (5,)
            Current robot state [x, y, theta, vx, omega]
        ref_trajectory : ndarray, shape (N, 7)
            Reference trajectory
        nearest_idx : int
            Index of nearest point on trajectory

        Returns
        -------
        ControlOutput
            Control commands
        """
        x, y, theta, vx, omega = state

        ref = ref_trajectory[nearest_idx]
        ref_vx = ref[3]
        ref_omega = ref[5]
        ref_kappa = ref[6]

        frenet_state = FrenetFrame.world_to_frenet(
            x, y, theta, vx, omega, ref
        )

        e = np.array([
            frenet_state.e_lat,
            frenet_state.e_yaw,
            frenet_state.e_v,
            frenet_state.e_omega,
        ])

        # Look-ahead feedforward: when the nearest reference point has near-
        # zero velocity (start/end ramp region), peek ahead on the trajectory
        # to find the velocity the robot should be ramping toward.  Errors are
        # still computed against the geometric nearest point.
        ref_vx_ff, ref_omega_ff = ref_vx, ref_omega
        if abs(ref_vx) < 0.01 and nearest_idx < len(ref_trajectory) - 10:
            la = min(nearest_idx + 30, len(ref_trajectory) - 1)
            ref_vx_ff = ref_trajectory[la, 3]
            ref_omega_ff = ref_trajectory[la, 5]

        # Floor the velocity used for linearization to avoid a degenerate
        # (uncontrollable) A matrix at near-zero speed.
        v_model = max(abs(ref_vx_ff), 0.05)

        kappa_changed = abs(ref_kappa - self._last_kappa) > 0.01
        v_changed = abs(v_model - self._last_v) > 0.05
        if self._K is None or kappa_changed or v_changed:
            A, B = self._build_system_matrices(v_model, ref_omega, ref_kappa)
            self._K = self._compute_gain_matrix(A, B)
            self._last_kappa = ref_kappa
            self._last_v = v_model

        delta = -self._K @ e

        v_cmd = float(np.clip(ref_vx_ff + delta[0], -self.max_v, self.max_v))
        omega_cmd = float(np.clip(ref_omega_ff + delta[1], -self.max_omega, self.max_omega))

        return ControlOutput(v_cmd=v_cmd, omega_cmd=omega_cmd)

    def get_info(self) -> dict:
        """Get controller information."""
        return {
            "controller": "LQR",
            "Q": np.diag(self.Q).tolist(),
            "R": np.diag(self.R).tolist(),
            "K": self._K.tolist() if self._K is not None else None,
            "tau_v": self.tau_v,
            "tau_omega": self.tau_omega,
        }
