"""Integration tests: full pipeline convergence verification.

Unlike unit tests that only check interfaces, these tests run the complete
trajectory → controller → kinematics → metrics pipeline and assert that
the tracking errors converge below concrete thresholds.
"""

import numpy as np
import pytest

from wheel_control.control import LQRController
from wheel_control.kinematics import DiffDriveKinematics
from wheel_control.simulation import SimulationEnv
from wheel_control.trajectory import (
    CircleTrajectory,
    SineTrajectory,
    StraightTrajectory,
    BezierSplineTrajectory,
)


@pytest.fixture
def robot():
    return DiffDriveKinematics(
        wheel_base=0.3,
        wheel_radius=0.05,
        max_v=1.5,
        max_omega=3.0,
        tau_v=0.1,
        tau_omega=0.08,
        v_acc_max=3.0,
        w_acc_max=10.0,
        dt=0.02,
    )


@pytest.fixture
def controller():
    return LQRController(
        dt=0.02,
        Q=[1.0, 2.0, 0.5, 0.5],
        R=[0.1, 0.1],
        tau_v=0.1,
        tau_omega=0.08,
        max_v=1.5,
        max_omega=3.0,
    )


def _run_tracking(robot, controller, trajectory, max_steps=2000, lateral_limit=1.0):
    """Run a tracking episode and return the result."""
    env = SimulationEnv(
        trajectory=trajectory,
        controller=controller,
        kinematics=robot,
        config={
            "dt": 0.02,
            "max_steps": max_steps,
            "lateral_limit": lateral_limit,
            "logging_enabled": False,
        },
    )
    return env.run_episode(seed=42)


class TestCircleTracking:
    def test_completes_without_diverging(self, robot, controller):
        gen = CircleTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.success, (
            f"Circle tracking failed: terminated={result.n_steps}, "
            f"final e_lat={result.final_errors['e_lat']:.4f}"
        )

    def test_lateral_error_bounded(self, robot, controller):
        gen = CircleTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.metrics["rms_lateral_error"] < 0.15, (
            f"Circle RMS lateral error too large: "
            f"{result.metrics['rms_lateral_error']:.4f}"
        )


class TestStraightTracking:
    def test_completes_without_diverging(self, robot, controller):
        gen = StraightTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.success

    def test_lateral_error_bounded(self, robot, controller):
        gen = StraightTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.metrics["rms_lateral_error"] < 0.10


class TestSineTracking:
    def test_completes_without_diverging(self, robot, controller):
        gen = SineTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.success

    def test_lateral_error_bounded(self, robot, controller):
        gen = SineTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.metrics["rms_lateral_error"] < 0.15


class TestBezierTracking:
    def test_completes_without_diverging(self, robot, controller):
        gen = BezierSplineTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.success

    def test_lateral_error_bounded(self, robot, controller):
        gen = BezierSplineTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.metrics["rms_lateral_error"] < 0.15


class TestMetricsConsistency:
    """Verify that metrics are internally consistent."""

    def test_rms_less_than_max(self, robot, controller):
        gen = CircleTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.metrics["rms_lateral_error"] <= result.metrics["max_lateral_error"]
        assert result.metrics["rms_yaw_error"] <= result.metrics["max_yaw_error"]

    def test_nonzero_steps(self, robot, controller):
        gen = StraightTrajectory(n_points=600, max_v=1.0, max_omega=3.0, max_acc=1.0)
        traj = gen.generate(np.random.default_rng(42))
        result = _run_tracking(robot, controller, traj)

        assert result.n_steps > 10
        assert len(result.step_data) == result.n_steps
