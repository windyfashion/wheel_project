"""Unit tests for simulation environment."""

import numpy as np
import pytest

from wheel_control.simulation import SimulationEnv, MetricEvaluator
from wheel_control.control import LQRController
from wheel_control.kinematics import DiffDriveKinematics
from wheel_control.trajectory import CircleTrajectory


class TestSimulationEnv:
    """Tests for SimulationEnv."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        gen = CircleTrajectory(n_points=100, max_v=1.0)
        trajectory = gen.generate()
        
        kinematics = DiffDriveKinematics(
            wheel_base=0.3,
            max_v=1.5,
            max_omega=3.0,
            dt=0.02,
        )
        
        controller = LQRController(dt=0.02, wheel_base=0.3)
        
        return SimulationEnv(
            trajectory=trajectory,
            controller=controller,
            kinematics=kinematics,
            config={"dt": 0.02, "max_steps": 200, "lateral_limit": 0.5},
        )
    
    def test_reset_initializes_state(self, env):
        """Test that reset initializes robot state."""
        state = env.reset()
        
        assert state.shape == (5,)
        assert np.isfinite(state).all()
    
    def test_step_returns_valid_output(self, env):
        """Test that step returns valid output."""
        env.reset()
        state, done, info = env.step()
        
        assert state.shape == (5,)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "e_lat" in info
        assert "e_yaw" in info
    
    def test_run_episode_returns_result(self, env):
        """Test that run_episode returns valid result."""
        result = env.run_episode()
        
        assert hasattr(result, "success")
        assert hasattr(result, "n_steps")
        assert hasattr(result, "metrics")
        assert result.n_steps > 0
    
    def test_get_tracking_errors(self, env):
        """Test getting tracking error history."""
        env.reset()
        
        for _ in range(10):
            env.step()
        
        errors = env.get_tracking_errors()
        
        assert "e_lat" in errors
        assert "e_yaw" in errors
        assert len(errors["e_lat"]) == 10


class TestMetricEvaluator:
    """Tests for MetricEvaluator."""
    
    def test_compute_basic_metrics(self):
        """Test basic metric computation."""
        evaluator = MetricEvaluator()
        
        e_lat = np.array([0.1, 0.2, 0.15, 0.1, 0.05])
        e_yaw = np.array([0.01, 0.02, 0.015, 0.01, 0.005])
        e_v = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        e_omega = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        metrics = evaluator.compute(e_lat, e_yaw, e_v, e_omega)
        
        assert "rms_lateral_error" in metrics
        assert "max_lateral_error" in metrics
        assert "rms_yaw_error" in metrics
        
        # Check RMS lateral error
        expected_rms = np.sqrt(np.mean(e_lat ** 2))
        assert np.isclose(metrics["rms_lateral_error"], expected_rms)
        
        # Check max lateral error
        assert np.isclose(metrics["max_lateral_error"], np.max(np.abs(e_lat)))
    
    def test_compute_from_steps(self):
        """Test metric computation from step list."""
        evaluator = MetricEvaluator()
        
        steps = [
            {"e_lat": 0.1, "e_yaw": 0.01, "e_v": 0.0, "e_omega": 0.0},
            {"e_lat": 0.2, "e_yaw": 0.02, "e_v": 0.0, "e_omega": 0.0},
            {"e_lat": 0.15, "e_yaw": 0.015, "e_v": 0.0, "e_omega": 0.0},
        ]
        
        metrics = evaluator.compute_from_steps(steps)
        
        assert metrics["rms_lateral_error"] > 0
        assert metrics["max_lateral_error"] == 0.2
    
    def test_format_metrics(self):
        """Test metric formatting."""
        evaluator = MetricEvaluator()
        
        metrics = {
            "rms_lateral_error": 0.1234,
            "max_lateral_error": 0.5678,
        }
        
        formatted = evaluator.format_metrics(metrics)
        
        assert "rms_lateral_error" in formatted
        assert "0.1234" in formatted
