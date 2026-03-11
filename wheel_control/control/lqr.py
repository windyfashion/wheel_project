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
    
    The system is linearized around the reference trajectory at each step.
    """
    
    def __init__(
        self,
        dt: float = 0.02,
        Q: np.ndarray | list | None = None,
        R: np.ndarray | list | None = None,
        wheel_base: float = 0.3,
    ):
        """Initialize LQR controller.
        
        Parameters
        ----------
        dt : float
            Control timestep
        Q : ndarray or list, shape (4,)
            State weight matrix diagonal [e_lat, e_yaw, e_v, e_omega]
        R : ndarray or list, shape (2,)
            Control weight matrix diagonal [delta_v, delta_omega]
        wheel_base : float
            Robot wheel base for kinematics
        """
        super().__init__(dt)
        
        # Default weight matrices
        if Q is None:
            Q = [1.0, 2.0, 0.5, 0.5]
        if R is None:
            R = [0.1, 0.1]
        
        self.Q = np.diag(Q) if isinstance(Q, list) else np.diag(Q)
        self.R = np.diag(R) if isinstance(R, list) else np.diag(R)
        self.wheel_base = wheel_base
        
        # Cache for K matrix
        self._K: np.ndarray | None = None
        self._last_kappa: float = 0.0
        
        # Internal state for integral term (optional)
        self._e_lat_integral: float = 0.0
        self._integral_gain: float = 0.0
    
    def reset(self) -> None:
        """Reset controller internal state."""
        self._K = None
        self._last_kappa = 0.0
        self._e_lat_integral = 0.0
    
    def set_weights(self, Q: np.ndarray | list, R: np.ndarray | list) -> None:
        """Set new weight matrices.
        
        Parameters
        ----------
        Q : ndarray or list
            State weight matrix diagonal
        R : ndarray or list
            Control weight matrix diagonal
        """
        self.Q = np.diag(Q) if isinstance(Q, list) else np.diag(Q)
        self.R = np.diag(R) if isinstance(R, list) else np.diag(R)
        self._K = None  # Force recomputation
    
    def _build_system_matrices(
        self,
        ref_v: float,
        ref_omega: float,
        kappa: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build linearized system matrices A and B.
        
        The linearized error dynamics in Frenet frame:
        e_lat_dot = ref_v * sin(e_yaw) + e_v * sin(e_yaw)
        e_yaw_dot = e_omega - ref_v * kappa * cos(e_yaw)
        e_v_dot = (v_cmd - v_actual) / tau  (handled by actuator dynamics)
        e_omega_dot = (omega_cmd - omega_actual) / tau  (handled by actuator dynamics)
        
        For small angles, sin(e_yaw) ~ e_yaw, cos(e_yaw) ~ 1.
        
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
            (A, B) system matrices
        """
        dt = self.dt
        
        # State: [e_lat, e_yaw, e_v, e_omega]
        # Control: [delta_v, delta_omega]
        
        A = np.array([
            # e_lat_dot derivatives
            [0, ref_v, 0, 0],           # de_lat/d(e_lat, e_yaw, e_v, e_omega)
            # e_yaw_dot derivatives
            [0, 0, 0, 1],               # de_yaw/d(e_lat, e_yaw, e_v, e_omega)
            # e_v_dot (actuator dynamics, simplified)
            [0, 0, 0, 0],               # de_v/d(...)
            # e_omega_dot (actuator dynamics, simplified)
            [0, 0, 0, 0],               # de_omega/d(...)
        ])
        
        B = np.array([
            # Control input effects
            [0, 0],                     # de_lat/d(delta_v, delta_omega)
            [0, 0],                     # de_yaw/d(delta_v, delta_omega)
            [1, 0],                     # de_v/d(delta_v, delta_omega)
            [0, 1],                     # de_omega/d(delta_v, delta_omega)
        ])
        
        # Discretize: A_d = I + A * dt, B_d = B * dt
        A_d = np.eye(4) + A * dt
        B_d = B * dt
        
        return A_d, B_d
    
    def _compute_gain_matrix(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Compute LQR gain matrix K using discrete-time Riccati equation.
        
        Solves: A'PA - P - A'PB(R + B'PB)^(-1)B'PA + Q = 0
        Then: K = (R + B'PB)^(-1)B'PA
        
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
            # Solve discrete-time algebraic Riccati equation
            P = linalg.solve_discrete_are(A, B, self.Q, self.R)
            
            # Compute K = (R + B'PB)^(-1)B'PA
            K = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
            
            return K
        except np.linalg.LinAlgError:
            # Fallback to previous K or default
            if self._K is None:
                self._K = np.array([
                    [0.5, 1.0, 0.3, 0.1],  # delta_v gains
                    [0.1, 0.5, 0.1, 0.3],  # delta_omega gains
                ])
            return self._K
    
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
        
        # Get reference point
        ref = ref_trajectory[nearest_idx]
        ref_x = ref[0]
        ref_y = ref[1]
        ref_yaw = ref[2]
        ref_vx = ref[3]
        ref_omega = ref[5]
        ref_kappa = ref[6]
        
        # Compute Frenet frame errors
        frenet_state = FrenetFrame.world_to_frenet(
            x, y, theta, vx, omega, ref
        )
        
        # State error vector
        e = np.array([
            frenet_state.e_lat,
            frenet_state.e_yaw,
            frenet_state.e_v,
            frenet_state.e_omega,
        ])
        
        # Build system matrices (can skip if kappa hasn't changed much)
        if self._K is None or abs(ref_kappa - self._last_kappa) > 0.01:
            A, B = self._build_system_matrices(ref_vx, ref_omega, ref_kappa)
            self._K = self._compute_gain_matrix(A, B)
            self._last_kappa = ref_kappa
        
        # Compute control: u = -K * e
        delta = -self._K @ e
        
        # Feedforward + feedback
        v_cmd = ref_vx + delta[0]
        omega_cmd = ref_omega + delta[1]
        
        return ControlOutput(v_cmd=float(v_cmd), omega_cmd=float(omega_cmd))
    
    def get_info(self) -> dict:
        """Get controller information."""
        return {
            "controller": "LQR",
            "Q": np.diag(self.Q).tolist(),
            "R": np.diag(self.R).tolist(),
            "K": self._K.tolist() if self._K is not None else None,
        }
