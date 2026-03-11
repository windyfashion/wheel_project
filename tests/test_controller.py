"""Unit tests for LQR controller."""

import numpy as np
import pytest

from wheel_control.control import LQRController
from wheel_control.trajectory import CircleTrajectory


class TestLQRController:
    """Tests for LQRController."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = LQRController(dt=0.02)
        
        assert controller.dt == 0.02
        assert controller._K is None
    
    def test_compute_control_returns_valid_output(self):
        """Test that compute_control returns valid output."""
        controller = LQRController(dt=0.02)
        
        # Generate simple trajectory
        gen = CircleTrajectory(n_points=100)
        trajectory = gen.generate()
        
        # Initial state at trajectory start
        state = np.array([
            trajectory[0, 0],  # x
            trajectory[0, 1],  # y
            trajectory[0, 2],  # theta
            0.0,               # vx
            0.0,               # omega
        ])
        
        output = controller.compute_control(state, trajectory, 0)
        
        assert isinstance(output.v_cmd, float)
        assert isinstance(output.omega_cmd, float)
    
    def test_reset_clears_internal_state(self):
        """Test that reset clears internal state."""
        controller = LQRController(dt=0.02)
        
        # Run some computation
        gen = CircleTrajectory(n_points=100)
        trajectory = gen.generate()
        state = np.zeros(5)
        controller.compute_control(state, trajectory, 0)
        
        # Reset
        controller.reset()
        
        assert controller._K is None
        assert controller._e_lat_integral == 0.0
    
    def test_set_weights_updates_matrices(self):
        """Test that set_weights updates Q and R matrices."""
        controller = LQRController(dt=0.02)
        
        new_Q = [2.0, 3.0, 1.0, 1.0]
        new_R = [0.2, 0.2]
        
        controller.set_weights(new_Q, new_R)
        
        assert np.allclose(np.diag(controller.Q), new_Q)
        assert np.allclose(np.diag(controller.R), new_R)
    
    def test_control_handles_large_errors(self):
        """Test that controller handles large errors gracefully."""
        controller = LQRController(dt=0.02)
        
        gen = CircleTrajectory(n_points=100)
        trajectory = gen.generate()
        
        # State far from trajectory
        state = np.array([100.0, 100.0, 0.0, 0.0, 0.0])
        
        output = controller.compute_control(state, trajectory, 0)
        
        # Should not produce NaN or Inf
        assert np.isfinite(output.v_cmd)
        assert np.isfinite(output.omega_cmd)
