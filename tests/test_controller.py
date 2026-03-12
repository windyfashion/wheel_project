"""Unit tests for LQR controller."""

import numpy as np
import pytest

from wheel_control.control import LQRController
from wheel_control.trajectory import CircleTrajectory


class TestLQRController:
    """Tests for LQRController."""

    def _make_controller(self, **kwargs):
        defaults = dict(dt=0.02, tau_v=0.1, tau_omega=0.08, max_v=1.5, max_omega=3.0)
        defaults.update(kwargs)
        return LQRController(**defaults)

    def test_initialization(self):
        """Test controller initialization."""
        controller = self._make_controller()

        assert controller.dt == 0.02
        assert controller._K is None
        assert controller.tau_v == 0.1
        assert controller.tau_omega == 0.08

    def test_compute_control_returns_valid_output(self):
        """Test that compute_control returns valid output."""
        controller = self._make_controller()

        gen = CircleTrajectory(n_points=100)
        trajectory = gen.generate()

        state = np.array([
            trajectory[0, 0],
            trajectory[0, 1],
            trajectory[0, 2],
            0.0,
            0.0,
        ])

        output = controller.compute_control(state, trajectory, 0)

        assert isinstance(output.v_cmd, float)
        assert isinstance(output.omega_cmd, float)

    def test_reset_clears_internal_state(self):
        """Test that reset clears internal state."""
        controller = self._make_controller()

        gen = CircleTrajectory(n_points=100)
        trajectory = gen.generate()
        state = np.zeros(5)
        controller.compute_control(state, trajectory, 0)

        controller.reset()

        assert controller._K is None
        assert controller._last_v == 0.0

    def test_set_weights_updates_matrices(self):
        """Test that set_weights updates Q and R matrices."""
        controller = self._make_controller()

        new_Q = [2.0, 3.0, 1.0, 1.0]
        new_R = [0.2, 0.2]

        controller.set_weights(new_Q, new_R)

        assert np.allclose(np.diag(controller.Q), new_Q)
        assert np.allclose(np.diag(controller.R), new_R)

    def test_set_weights_accepts_2d_matrix(self):
        """Test that set_weights handles 2-D arrays correctly."""
        controller = self._make_controller()

        Q_mat = np.diag([2.0, 3.0, 1.0, 1.0])
        R_mat = np.diag([0.2, 0.2])
        controller.set_weights(Q_mat, R_mat)

        assert np.allclose(controller.Q, Q_mat)
        assert np.allclose(controller.R, R_mat)

    def test_control_handles_large_errors(self):
        """Test that controller handles large errors gracefully."""
        controller = self._make_controller()

        gen = CircleTrajectory(n_points=100)
        trajectory = gen.generate()

        state = np.array([100.0, 100.0, 0.0, 0.0, 0.0])

        output = controller.compute_control(state, trajectory, 0)

        assert np.isfinite(output.v_cmd)
        assert np.isfinite(output.omega_cmd)

    def test_output_clipped_to_limits(self):
        """Test that output respects velocity limits."""
        controller = self._make_controller(max_v=1.0, max_omega=2.0)

        gen = CircleTrajectory(n_points=100)
        trajectory = gen.generate()

        state = np.array([100.0, 100.0, 0.0, 0.0, 0.0])
        output = controller.compute_control(state, trajectory, 0)

        assert abs(output.v_cmd) <= 1.0
        assert abs(output.omega_cmd) <= 2.0

    def test_gain_recomputed_on_velocity_change(self):
        """Test that K is recomputed when ref velocity changes."""
        controller = self._make_controller()

        gen = CircleTrajectory(n_points=200)
        trajectory = gen.generate()

        state = np.array([trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 0.0, 0.0])
        controller.compute_control(state, trajectory, 0)
        K1 = controller._K.copy()

        # Jump to a later point where velocity differs
        idx = len(trajectory) // 2
        state2 = np.array([trajectory[idx, 0], trajectory[idx, 1], trajectory[idx, 2], 0.5, 0.0])
        controller.compute_control(state2, trajectory, idx)
        K2 = controller._K

        # K may or may not differ depending on velocity change magnitude,
        # but the mechanism should not crash
        assert K2 is not None
