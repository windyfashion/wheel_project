"""Base classes for planners."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PathPlannerBase(ABC):
    """Base class for path planners.
    
    Path planners generate geometric paths (x, y coordinates).
    """
    
    @abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Plan a path from start to goal.
        
        Parameters
        ----------
        start : ndarray, shape (2,) or (3,)
            Start position [x, y] or [x, y, theta]
        goal : ndarray, shape (2,) or (3,)
            Goal position [x, y] or [x, y, theta]
        
        Returns
        -------
        ndarray, shape (N, 2)
            Path waypoints [(x, y), ...]
        """
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """Set planner parameters."""


class VelocityPlannerBase(ABC):
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
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """Set planner parameters."""
