"""Abstract base class for controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ControlOutput:
    """Controller output."""
    v_cmd: float      # Linear velocity command
    omega_cmd: float  # Angular velocity command


class ControllerBase(ABC):
    """Base class for all trajectory tracking controllers.
    
    All controllers must implement the compute_control method that takes
    the current state and reference trajectory, and returns control commands.
    """
    
    def __init__(self, dt: float = 0.02):
        """Initialize controller.
        
        Parameters
        ----------
        dt : float
            Control timestep
        """
        self.dt = dt
    
    @abstractmethod
    def compute_control(
        self,
        state: np.ndarray,
        ref_trajectory: np.ndarray,
        nearest_idx: int,
    ) -> ControlOutput:
        """Compute control commands.
        
        Parameters
        ----------
        state : ndarray, shape (5,)
            Current robot state [x, y, theta, vx, omega]
        ref_trajectory : ndarray, shape (N, 7)
            Reference trajectory [x, y, yaw, vx, vy, omega, kappa]
        nearest_idx : int
            Index of nearest point on trajectory
        
        Returns
        -------
        ControlOutput
            Control commands (v_cmd, omega_cmd)
        """
    
    @abstractmethod
    def reset(self) -> None:
        """Reset controller internal state."""
    
    def get_info(self) -> dict:
        """Get controller information for logging.
        
        Returns
        -------
        dict
            Controller-specific information
        """
        return {}
