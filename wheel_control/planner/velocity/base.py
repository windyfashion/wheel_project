"""Base class for velocity planners."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np


class VelocityPlannerBase:
    """Base class for velocity planners.
    
    Velocity planners assign speed profiles to geometric paths.
    """
    
    @abstractmethod
    def plan(
        self,
        path: np.ndarray,
        max_v: float,
        max_omega: float,
        max_acc: float,
    ) -> np.ndarray:
        """Plan velocity profile for a path.
        
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
    
    def set_parameters(self, **kwargs) -> None:
        """Set planner parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
