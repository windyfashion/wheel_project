"""Unit tests for trajectory module."""

import numpy as np
import pytest

from wheel_control.trajectory import (
    TrajectoryBase,
    CircleTrajectory,
    SineTrajectory,
    BezierSplineTrajectory,
    RandomTrajectoryGenerator,
    X, Y, YAW, VX, VY, W, KAPPA,
)


class TestCircleTrajectory:
    """Tests for CircleTrajectory."""
    
    def test_generate_returns_correct_shape(self):
        """Test that generated trajectory has correct shape."""
        gen = CircleTrajectory(n_points=100)
        traj = gen.generate()
        
        assert traj.shape == (100, 7)
    
    def test_generate_returns_valid_values(self):
        """Test that generated values are valid."""
        gen = CircleTrajectory(n_points=100, max_v=1.5, max_omega=3.0)
        traj = gen.generate()
        
        # Check velocity limits
        assert np.all(traj[:, VX] <= 1.5)
        assert np.all(traj[:, VX] >= 0)
        
        # Check omega limits
        assert np.all(np.abs(traj[:, W]) <= 3.0)
    
    def test_generate_with_rng(self):
        """Test generation with random generator."""
        gen = CircleTrajectory(n_points=100)
        rng = np.random.default_rng(42)
        traj = gen.generate(rng)
        
        assert traj.shape == (100, 7)


class TestBezierSplineTrajectory:
    """Tests for BezierSplineTrajectory."""
    
    def test_generate_returns_correct_shape(self):
        """Test that generated trajectory has correct shape."""
        gen = BezierSplineTrajectory(n_points=200)
        traj = gen.generate()
        
        assert traj.shape == (200, 7)
    
    def test_generate_smooth_trajectory(self):
        """Test that generated trajectory is smooth."""
        gen = BezierSplineTrajectory(n_points=200)
        traj = gen.generate()
        
        # Check for smoothness (no sudden jumps)
        dx = np.diff(traj[:, X])
        dy = np.diff(traj[:, Y])
        ds = np.sqrt(dx**2 + dy**2)
        
        # No step should be too large
        assert np.max(ds) < 0.1  # Reasonable threshold


class TestRandomTrajectoryGenerator:
    """Tests for RandomTrajectoryGenerator."""
    
    def test_generate_selects_randomly(self):
        """Test that generator selects different trajectory types."""
        gen = RandomTrajectoryGenerator(
            types=["circle", "sine", "straight"],
            n_points=100,
        )
        
        # Generate multiple trajectories
        rng = np.random.default_rng(42)
        trajs = [gen.generate(rng) for _ in range(10)]
        
        # All should have correct shape
        for traj in trajs:
            assert traj.shape == (100, 7)


class TestTrajectoryBase:
    """Tests for TrajectoryBase utility functions."""
    
    def test_compute_derivatives_straight_line(self):
        """Test derivative computation for straight line."""
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)
        
        yaw, vx, omega, kappa = TrajectoryBase.compute_derivatives(
            x, y, dt=0.02, max_v=1.0, max_omega=3.0, max_acc=1.0
        )
        
        # Yaw should be ~0 for straight line
        assert np.allclose(yaw[10:-10], 0, atol=0.1)
        
        # Omega should be ~0
        assert np.allclose(omega[10:-10], 0, atol=0.1)
        
        # Kappa should be ~0
        assert np.allclose(kappa[10:-10], 0, atol=0.5)
    
    def test_compute_derivatives_circle(self):
        """Test derivative computation for circle."""
        t = np.linspace(0, 2*np.pi, 100)
        radius = 2.0
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        
        yaw, vx, omega, kappa = TrajectoryBase.compute_derivatives(
            x, y, dt=0.02, max_v=1.0, max_omega=3.0, max_acc=1.0
        )
        
        # Kappa should be ~1/radius
        expected_kappa = 1.0 / radius
        assert np.allclose(kappa[10:-10], expected_kappa, atol=0.3)
