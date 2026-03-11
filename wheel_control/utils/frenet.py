"""Frenet frame coordinate transformation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..trajectory.base import X, Y, YAW, VX, W, KAPPA


@dataclass
class FrenetState:
    """State in Frenet coordinate frame."""
    e_s: float      # longitudinal error (along path)
    e_lat: float    # lateral error (perpendicular to path)
    e_yaw: float    # heading error
    e_v: float      # velocity error
    e_omega: float  # angular velocity error


class FrenetFrame:
    """Frenet coordinate frame utilities for trajectory tracking.
    
    The Frenet frame is a moving coordinate system attached to the reference
    trajectory. It decomposes errors into:
    - Longitudinal (along the path tangent)
    - Lateral (perpendicular to the path tangent)
    """

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def world_to_frenet(
        x: float,
        y: float,
        theta: float,
        vx: float,
        omega: float,
        ref_point: np.ndarray,
    ) -> FrenetState:
        """Transform world coordinates to Frenet frame.
        
        Parameters
        ----------
        x, y : float
            Robot position in world frame
        theta : float
            Robot heading in world frame
        vx, omega : float
            Robot velocities
        ref_point : ndarray, shape (7,)
            Reference point [x, y, yaw, vx, vy, omega, kappa]
        
        Returns
        -------
        FrenetState
            Errors in Frenet frame
        """
        # Position error in world frame
        dx = x - ref_point[X]
        dy = y - ref_point[Y]
        
        # Transform to Frenet frame using reference heading
        ref_yaw = ref_point[YAW]
        cos_yaw = np.cos(ref_yaw)
        sin_yaw = np.sin(ref_yaw)
        
        # Longitudinal and lateral errors
        e_s = cos_yaw * dx + sin_yaw * dy
        e_lat = -sin_yaw * dx + cos_yaw * dy
        
        # Heading error
        e_yaw = FrenetFrame.wrap_angle(theta - ref_yaw)
        
        # Velocity errors
        e_v = vx - ref_point[VX]
        e_omega = omega - ref_point[W]
        
        return FrenetState(
            e_s=e_s,
            e_lat=e_lat,
            e_yaw=e_yaw,
            e_v=e_v,
            e_omega=e_omega,
        )

    @staticmethod
    def frenet_to_world(
        e_s: float,
        e_lat: float,
        e_yaw: float,
        ref_point: np.ndarray,
    ) -> tuple[float, float, float]:
        """Transform Frenet frame to world coordinates.
        
        Parameters
        ----------
        e_s, e_lat : float
            Longitudinal and lateral errors in Frenet frame
        e_yaw : float
            Heading error
        ref_point : ndarray, shape (7,)
            Reference point [x, y, yaw, vx, vy, omega, kappa]
        
        Returns
        -------
        tuple[float, float, float]
            (x, y, theta) in world frame
        """
        ref_yaw = ref_point[YAW]
        cos_yaw = np.cos(ref_yaw)
        sin_yaw = np.sin(ref_yaw)
        
        # Transform position from Frenet to world
        x = ref_point[X] + cos_yaw * e_s - sin_yaw * e_lat
        y = ref_point[Y] + sin_yaw * e_s + cos_yaw * e_lat
        theta = FrenetFrame.wrap_angle(ref_yaw + e_yaw)
        
        return x, y, theta

    @staticmethod
    def find_nearest_point(
        x: float,
        y: float,
        trajectory: np.ndarray,
        start_idx: int = 0,
        search_range: int | None = None,
    ) -> int:
        """Find the nearest point on the trajectory to the robot.
        
        Parameters
        ----------
        x, y : float
            Robot position
        trajectory : ndarray, shape (N, 7)
            Reference trajectory
        start_idx : int
            Starting index for search (for efficiency)
        search_range : int, optional
            Number of points to search forward
        
        Returns
        -------
        int
            Index of nearest point on trajectory
        """
        n_points = len(trajectory)
        
        if search_range is None:
            search_range = n_points
        
        end_idx = min(start_idx + search_range, n_points)
        
        # Compute distances to all candidate points
        dx = trajectory[start_idx:end_idx, X] - x
        dy = trajectory[start_idx:end_idx, Y] - y
        distances = dx**2 + dy**2
        
        # Find minimum
        local_min_idx = int(np.argmin(distances))
        return start_idx + local_min_idx

    @staticmethod
    def find_nearest_point_with_progress(
        x: float,
        y: float,
        trajectory: np.ndarray,
        prev_idx: int,
    ) -> int:
        """Find nearest point ensuring monotonic progress along trajectory.
        
        This method ensures the robot doesn't "go backwards" on the trajectory
        by only searching forward from the previous nearest point.
        
        Parameters
        ----------
        x, y : float
            Robot position
        trajectory : ndarray, shape (N, 7)
            Reference trajectory
        prev_idx : int
            Previous nearest point index
        
        Returns
        -------
        int
            Index of nearest point (guaranteed >= prev_idx)
        """
        n_points = len(trajectory)
        
        # Search forward from previous index
        search_start = prev_idx
        search_end = min(prev_idx + 50, n_points)  # Look ahead 50 points
        
        dx = trajectory[search_start:search_end, X] - x
        dy = trajectory[search_start:search_end, Y] - y
        distances = dx**2 + dy**2
        
        local_min_idx = int(np.argmin(distances))
        return search_start + local_min_idx
