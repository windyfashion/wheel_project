"""Simulation environment for trajectory tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..control.base import ControllerBase
from ..kinematics import DiffDriveKinematics
from ..trajectory.base import X, Y, YAW, VX, W, KAPPA
from ..utils.frenet import FrenetFrame
from ..utils.logger import ControlLogger, StepData


@dataclass
class EpisodeResult:
    """Result of a simulation episode."""
    success: bool
    n_steps: int
    final_errors: dict[str, float]
    metrics: dict[str, float]
    trajectory: np.ndarray
    actual_path: np.ndarray
    step_data: list[StepData] = field(default_factory=list)


class SimulationEnv:
    """Simulation environment for trajectory tracking control.

    Example
    -------
    >>> env = SimulationEnv(trajectory, controller, kinematics, config)
    >>> result = env.run_episode()
    >>> print(f"RMS lateral error: {result.metrics['rms_lateral_error']}")
    """

    def __init__(
        self,
        trajectory: np.ndarray,
        controller: ControllerBase,
        kinematics: DiffDriveKinematics,
        config: dict | None = None,
    ):
        self.trajectory = trajectory
        self.controller = controller
        self.kinematics = kinematics

        self.config = config or {}
        self.dt = self.config.get("dt", 0.02)
        self.max_steps = self.config.get("max_steps", 1000)
        self.lateral_limit = self.config.get("lateral_limit", 0.5)
        self.init_noise = self.config.get("init_noise", 0.1)
        self._logging_enabled = self.config.get("logging_enabled", True)

        self._step_count = 0
        self._nearest_idx = 0
        self._actual_path: list[np.ndarray] = []
        self._step_data: list[StepData] = []

        if self._logging_enabled:
            log_dir = self.config.get("log_dir", "logs")
            self.logger = ControlLogger(log_dir)
        else:
            self.logger = None

    def reset(
        self,
        init_state: np.ndarray | None = None,
        noise: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Reset simulation to initial state.

        Parameters
        ----------
        init_state : ndarray, optional
            Initial state [x, y, theta, vx, omega]. If None, uses trajectory start.
        noise : float, optional
            Position noise magnitude. Overrides config.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ndarray
            Initial state
        """
        self._step_count = 0
        self._nearest_idx = 0
        self._actual_path = []
        self._step_data = []

        self.controller.reset()

        if noise is None:
            noise = self.init_noise

        if init_state is None:
            rng = np.random.default_rng(seed)
            self.kinematics.reset(
                x=self.trajectory[0, X] + rng.uniform(-noise, noise),
                y=self.trajectory[0, Y] + rng.uniform(-noise, noise),
                theta=self.trajectory[0, YAW] + rng.uniform(-noise * 0.5, noise * 0.5),
                vx=0.0,
                omega=0.0,
            )
        else:
            self.kinematics.reset(
                x=init_state[0],
                y=init_state[1],
                theta=init_state[2],
                vx=init_state[3] if len(init_state) > 3 else 0.0,
                omega=init_state[4] if len(init_state) > 4 else 0.0,
            )

        if self.logger is not None:
            self.logger.start_episode()

        return self.kinematics.state

    def step(self) -> tuple[np.ndarray, bool, dict]:
        """Execute one simulation step.

        Returns
        -------
        tuple[ndarray, bool, dict]
            (state, done, info)
        """
        state = self.kinematics.state
        x, y, theta, vx, omega = state

        self._nearest_idx = FrenetFrame.find_nearest_point_with_progress(
            x, y, self.trajectory, self._nearest_idx
        )

        control_output = self.controller.compute_control(
            state, self.trajectory, self._nearest_idx
        )

        self.kinematics.step(control_output.v_cmd, control_output.omega_cmd)

        new_state = self.kinematics.state
        ref = self.trajectory[self._nearest_idx]

        frenet = FrenetFrame.world_to_frenet(
            new_state[0], new_state[1], new_state[2],
            new_state[3], new_state[4], ref
        )

        self._actual_path.append(new_state[:3].copy())

        step_data = StepData(
            step=self._step_count,
            time=self._step_count * self.dt,
            x=new_state[0], y=new_state[1], theta=new_state[2],
            vx=new_state[3], omega=new_state[4],
            ref_x=ref[X], ref_y=ref[Y], ref_yaw=ref[YAW],
            ref_vx=ref[VX], ref_omega=ref[W], ref_kappa=ref[KAPPA],
            e_lat=frenet.e_lat, e_yaw=frenet.e_yaw,
            e_v=frenet.e_v, e_omega=frenet.e_omega,
            v_cmd=control_output.v_cmd, omega_cmd=control_output.omega_cmd,
        )
        self._step_data.append(step_data)
        if self.logger is not None:
            self.logger.log_step(step_data)

        self._step_count += 1

        terminated = abs(frenet.e_lat) > self.lateral_limit
        finished = self._nearest_idx >= len(self.trajectory) - 1
        truncated = self._step_count >= self.max_steps

        done = terminated or finished or truncated

        info = {
            "e_lat": frenet.e_lat,
            "e_yaw": frenet.e_yaw,
            "nearest_idx": self._nearest_idx,
            "terminated": terminated,
            "finished": finished,
            "truncated": truncated,
        }

        return new_state, done, info

    def run_episode(
        self,
        init_state: np.ndarray | None = None,
        progress_callback: Any | None = None,
        seed: int | None = None,
    ) -> EpisodeResult:
        """Run a complete episode.

        Parameters
        ----------
        init_state : ndarray, optional
            Initial state
        progress_callback : callable, optional
            Callback function(step, state, done, info)
        seed : int, optional
            Random seed for reproducible initial noise.

        Returns
        -------
        EpisodeResult
            Episode result with metrics
        """
        from ..simulation.metrics import MetricEvaluator

        self.reset(init_state, seed=seed)

        done = False
        info: dict = {}
        while not done:
            state, done, info = self.step()

            if progress_callback:
                progress_callback(self._step_count, state, done, info)

        final_errors = {
            "e_lat": info.get("e_lat", 0.0),
            "e_yaw": info.get("e_yaw", 0.0),
        }

        evaluator = MetricEvaluator()
        metrics = evaluator.compute_from_steps([
            {
                "e_lat": s.e_lat,
                "e_yaw": s.e_yaw,
                "e_v": s.e_v,
                "e_omega": s.e_omega,
            }
            for s in self._step_data
        ])

        if self.logger is not None:
            self.logger.end_episode()

        return EpisodeResult(
            success=info.get("finished", False) and not info.get("terminated", False),
            n_steps=self._step_count,
            final_errors=final_errors,
            metrics=metrics,
            trajectory=self.trajectory.copy(),
            actual_path=np.array(self._actual_path) if self._actual_path else np.empty((0, 3)),
            step_data=self._step_data.copy(),
        )

    def get_current_state(self) -> np.ndarray:
        """Get current robot state."""
        return self.kinematics.state

    def get_tracking_errors(self) -> dict[str, list[float]]:
        """Get history of tracking errors."""
        return {
            "e_lat": [s.e_lat for s in self._step_data],
            "e_yaw": [s.e_yaw for s in self._step_data],
            "e_v": [s.e_v for s in self._step_data],
            "e_omega": [s.e_omega for s in self._step_data],
        }

    def export_log(self, path: str | None = None) -> str:
        """Export episode log to CSV.

        Parameters
        ----------
        path : str, optional
            Output path

        Returns
        -------
        str
            Path to exported file
        """
        if self.logger is not None:
            return str(self.logger.export_csv(path, data=self._step_data))
        if not self._step_data:
            raise ValueError("No episode data to export")
        from ..utils.logger import ControlLogger
        tmp_logger = ControlLogger("logs")
        return str(tmp_logger.export_csv(path, data=self._step_data))
