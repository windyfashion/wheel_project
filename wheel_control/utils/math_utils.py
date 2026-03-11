"""Mathematical utilities for control."""

from __future__ import annotations

import numpy as np


class MathUtils:
    """Collection of mathematical utility functions."""
    
    @staticmethod
    def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
        """Wrap angle(s) to [-pi, pi].
        
        Parameters
        ----------
        angle : float or ndarray
            Angle(s) in radians
        
        Returns
        -------
        float or ndarray
            Wrapped angle(s)
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """Compute shortest angular difference a - b.
        
        Parameters
        ----------
        a, b : float
            Angles in radians
        
        Returns
        -------
        float
            Angular difference in [-pi, pi]
        """
        return MathUtils.wrap_angle(a - b)
    
    @staticmethod
    def linear_interpolate(
        x: float,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
    ) -> float:
        """Linear interpolation.
        
        Parameters
        ----------
        x : float
            Query point
        x_arr : ndarray
            X coordinates (must be sorted)
        y_arr : ndarray
            Y coordinates
        
        Returns
        -------
        float
            Interpolated value
        """
        if x <= x_arr[0]:
            return float(y_arr[0])
        if x >= x_arr[-1]:
            return float(y_arr[-1])
        
        idx = np.searchsorted(x_arr, x) - 1
        t = (x - x_arr[idx]) / (x_arr[idx + 1] - x_arr[idx])
        return float(y_arr[idx] + t * (y_arr[idx + 1] - y_arr[idx]))
    
    @staticmethod
    def compute_curvature(
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute curvature of a path.
        
        Parameters
        ----------
        x, y : ndarray
            Path coordinates
        
        Returns
        -------
        ndarray
            Curvature at each point
        """
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2) ** 1.5 + 1e-12
        
        return numerator / denominator
    
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize a vector.
        
        Parameters
        ----------
        v : ndarray
            Input vector
        
        Returns
        -------
        ndarray
            Normalized vector
        """
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    
    @staticmethod
    def rotation_matrix(theta: float) -> np.ndarray:
        """Create 2D rotation matrix.
        
        Parameters
        ----------
        theta : float
            Rotation angle in radians
        
        Returns
        -------
        ndarray, shape (2, 2)
            Rotation matrix
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])
    
    @staticmethod
    def smooth_signal(
        signal: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """Smooth a signal using moving average.
        
        Parameters
        ----------
        signal : ndarray
            Input signal
        window_size : int
            Size of smoothing window
        
        Returns
        -------
        ndarray
            Smoothed signal
        """
        if window_size < 2:
            return signal
        
        kernel = np.ones(window_size) / window_size
        padded = np.pad(signal, window_size // 2, mode="edge")
        return np.convolve(padded, kernel, mode="valid")
