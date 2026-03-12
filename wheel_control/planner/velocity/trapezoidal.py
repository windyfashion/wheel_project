"""Trapezoidal velocity profile planner."""

from __future__ import annotations

import numpy as np

from ...utils.math_utils import MathUtils


class TrapezoidalVelocityPlanner:
    """Velocity planner using trapezoidal profile.
    
    Generates a velocity profile with:
    - Acceleration phase
    - Constant velocity phase (cruise)
    - Deceleration phase
    
    The profile respects maximum velocity, acceleration, and
    curvature constraints.
    """
    
    def __init__(
        self,
        start_velocity: float = 0.0,
        end_velocity: float = 0.0,
    ):
        """Initialize trapezoidal velocity planner.
        
        Parameters
        ----------
        start_velocity : float
            Initial velocity
        end_velocity : float
            Final velocity
        """
        self.start_velocity = start_velocity
        self.end_velocity = end_velocity
    
    def plan(
        self,
        path: np.ndarray,
        max_v: float,
        max_omega: float,
        max_acc: float,
    ) -> np.ndarray:
        """Plan trapezoidal velocity profile.
        
        Parameters
        ----------
        path : ndarray, shape (N, 2)
            Path waypoints
        max_v : float
            Maximum linear velocity
        max_omega : float
            Maximum angular velocity
        max_acc : float
            Maximum acceleration
        
        Returns
        -------
        ndarray, shape (N,)
            Velocity at each waypoint
        """
        n_points = len(path)
        if n_points < 2:
            return np.array([self.start_velocity])
        
        # Compute path curvature
        kappa = self._compute_curvature(path)
        
        # Curvature-limited velocity
        v_kappa = np.where(np.abs(kappa) > 1e-6, max_omega / np.abs(kappa), max_v)
        v_limit = np.minimum(max_v, v_kappa)
        
        # Compute arc length
        ds = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
        arc = np.concatenate([[0], np.cumsum(ds)])
        total_length = arc[-1]
        
        # Forward pass (acceleration limited)
        v_forward = np.zeros(n_points)
        v_forward[0] = self.start_velocity
        
        for i in range(1, n_points):
            # Maximum velocity at this point from acceleration
            v_from_prev = np.sqrt(
                v_forward[i - 1] ** 2 + 2 * max_acc * ds[i - 1]
            )
            v_forward[i] = min(v_from_prev, v_limit[i])
        
        # Backward pass (deceleration limited)
        v_backward = np.zeros(n_points)
        v_backward[-1] = self.end_velocity
        
        for i in range(n_points - 2, -1, -1):
            v_from_next = np.sqrt(
                v_backward[i + 1] ** 2 + 2 * max_acc * ds[i]
            )
            v_backward[i] = min(v_from_next, v_limit[i])
        
        # Take minimum of forward and backward
        v_profile = np.minimum(v_forward, v_backward)
        
        # Smooth the profile slightly to reduce jerk
        v_profile = self._smooth_profile(v_profile, window=3)
        
        return v_profile
    
    @staticmethod
    def _compute_curvature(path: np.ndarray) -> np.ndarray:
        """Compute curvature along path.

        Parameters
        ----------
        path : ndarray, shape (N, 2)
            Path waypoints

        Returns
        -------
        ndarray, shape (N,)
            Curvature at each point
        """
        if len(path) < 3:
            return np.zeros(len(path))
        return MathUtils.compute_curvature(path[:, 0], path[:, 1])
    
    def _smooth_profile(self, v: np.ndarray, window: int = 3) -> np.ndarray:
        """Smooth velocity profile with moving average.
        
        Parameters
        ----------
        v : ndarray
            Velocity profile
        window : int
            Smoothing window size
        
        Returns
        -------
        ndarray
            Smoothed profile
        """
        if window < 2:
            return v
        
        kernel = np.ones(window) / window
        padded = np.pad(v, window // 2, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        
        # Ensure we don't exceed original limits
        return np.minimum(smoothed, v)
    
    def plan_with_timing(
        self,
        path: np.ndarray,
        max_v: float,
        max_omega: float,
        max_acc: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan velocity profile and compute timestamps.
        
        Parameters
        ----------
        path : ndarray, shape (N, 2)
            Path waypoints
        max_v, max_omega, max_acc : float
            Velocity and acceleration limits
        dt : float
            Timestep
        
        Returns
        -------
        tuple[ndarray, ndarray]
            (velocity_profile, timestamps)
        """
        v_profile = self.plan(path, max_v, max_omega, max_acc)
        
        # Compute arc length
        ds = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
        arc = np.concatenate([[0], np.cumsum(ds)])
        
        # Compute time at each point
        times = np.zeros(len(path))
        for i in range(1, len(path)):
            if v_profile[i - 1] > 0:
                times[i] = times[i - 1] + ds[i - 1] / v_profile[i - 1]
            else:
                times[i] = times[i - 1] + dt
        
        return v_profile, times
