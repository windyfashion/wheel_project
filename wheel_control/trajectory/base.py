"""Abstract base class for trajectory generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# Column indices for the trajectory array
X, Y, YAW, VX, VY, W, KAPPA = range(7)


class TrajectoryBase(ABC):
    """Base class for trajectory generators.

    All generators must produce an ndarray of shape (N, 7) with columns:
        [x, y, yaw, vx, vy, omega, kappa]

    For differential drive robots, vy is always 0.
    """

    @abstractmethod
    def generate(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a trajectory.

        Parameters
        ----------
        rng : numpy random Generator, optional
            Used for randomized trajectory parameters.

        Returns
        -------
        ndarray, shape (N, 7)
        """

    @staticmethod
    def compute_derivatives(
        x: np.ndarray,
        y: np.ndarray,
        dt: float,
        max_v: float,
        max_omega: float,
        max_acc: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute yaw, speed, omega, and curvature from x/y waypoints.

        Physics-consistent pipeline
        ----------------------------
        1. **Geometry**: yaw via ``np.unwrap(arctan2)`` — no 2pi pseudo-pulses;
           curvature from the parametric formula ``k = (x'y'' - y'x'') / |r'|^3``.
        2. **Speed profile**: first limited by curvature (``v <= max_w / |k|``),
           then smoothed with a forward-backward acceleration pass so that
           ``|Dv^2| <= 2*a_max*Ds`` between every pair of adjacent waypoints.
           Boundary: ``v[0] = v[-1] = 0`` (start / stop from rest).
        3. **Omega**: derived from the coupling ``w = k*v`` — never clipped
           independently, so the non-holonomic constraint is always respected.

        Parameters
        ----------
        x, y : ndarray, shape (N,)
        dt : float              (kept for API compat; unused internally)
        max_v : float           max linear speed  (m/s)
        max_omega : float       max angular speed  (rad/s)
        max_acc : float         max tangential acceleration  (m/s^2)

        Returns (yaw, vx, omega, kappa), each shape (N,).
        """
        N = len(x)

        # ---- 1. Geometry (index-parameterised central differences) ----
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        yaw = np.unwrap(np.arctan2(dy, dx))

        speed_param = np.sqrt(dx**2 + dy**2)
        kappa = (dx * ddy - dy * ddx) / (speed_param**3 + 1e-12)

        # Arc-length increments between consecutive waypoints
        ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        ds = np.maximum(ds, 1e-12)

        # ---- 2. Curvature-limited speed ceiling ----
        v_limit = np.minimum(max_v, max_omega / (np.abs(kappa) + 1e-9))

        # ---- 3. Forward-backward acceleration pass (v^2 kinematics) ----
        def _fwd_bwd(v: np.ndarray) -> np.ndarray:
            v[0] = 0.0
            for i in range(1, N):
                v_fwd = np.sqrt(v[i - 1] ** 2 + 2.0 * max_acc * ds[i - 1])
                if v_fwd < v[i]:
                    v[i] = v_fwd
            v[-1] = 0.0
            for i in range(N - 2, -1, -1):
                v_bwd = np.sqrt(v[i + 1] ** 2 + 2.0 * max_acc * ds[i])
                if v_bwd < v[i]:
                    v[i] = v_bwd
            return v

        vx = _fwd_bwd(v_limit.copy())

        # ---- 4. Gaussian smoothing to soften jerk at phase boundaries ----
        #   The trapezoidal profile from step 3 has discontinuous acceleration
        #   (infinite jerk) at accel/cruise/decel transitions.  A small Gaussian
        #   kernel rounds these corners, then a second fwd-bwd pass guarantees
        #   that no constraint is violated after smoothing.
        sigma = max(1, N // 60)
        radius = int(3.0 * sigma + 0.5)
        k = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (k / sigma) ** 2)
        kernel /= kernel.sum()
        vx = np.convolve(np.pad(vx, radius, mode="edge"), kernel, mode="valid")

        vx = np.minimum(vx, v_limit)
        np.maximum(vx, 0.0, out=vx)
        vx = _fwd_bwd(vx)

        # ---- 5. w = k * v  (kinematic coupling, never clipped alone) ----
        omega = kappa * vx

        return yaw, vx, omega, kappa
