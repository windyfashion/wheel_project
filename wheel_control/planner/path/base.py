"""Base class for path planners."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np


class PathPlannerBase:
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
    
    def set_parameters(self, **kwargs) -> None:
        """Set planner parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Parameter name-value pairs
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
