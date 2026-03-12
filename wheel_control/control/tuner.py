"""LQR parameter auto-tuning module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lqr import LQRController
from ..kinematics import DiffDriveKinematics
from ..simulation.env import SimulationEnv


@dataclass
class TuningResult:
    """Result of parameter tuning."""
    best_Q: list[float]
    best_R: list[float]
    best_metric: float
    all_trials: list[dict]


class LQRTuner:
    """LQR parameter auto-tuning using optimization.

    Supports multiple optimization methods:
    - bayesian: Bayesian optimization (requires scikit-optimize)
    - grid: Grid search
    - random: Random search

    Example
    -------
    >>> tuner = LQRTuner(controller, kinematics, method="random")
    >>> result = tuner.tune(trajectory, n_trials=50)
    >>> controller.set_weights(result.best_Q, result.best_R)
    """

    def __init__(
        self,
        controller: LQRController,
        kinematics: DiffDriveKinematics,
        method: str = "bayesian",
        metric: str = "rms_lateral_error",
    ):
        self.controller = controller
        self.kinematics = kinematics
        self.method = method
        self.metric = metric

        self._trial_history: list[dict] = []

    def objective(
        self,
        params: dict,
        trajectory: np.ndarray,
        dt: float,
        max_steps: int,
    ) -> float:
        """Evaluate parameter set on trajectory.

        Parameters
        ----------
        params : dict
            Parameters with keys "Q" (list of 4) and "R" (list of 2)
        trajectory : ndarray
            Reference trajectory
        dt : float
            Timestep
        max_steps : int
            Maximum simulation steps

        Returns
        -------
        float
            Metric value to minimize
        """
        self.controller.set_weights(params["Q"], params["R"])
        self.controller.reset()

        env = SimulationEnv(
            trajectory=trajectory,
            controller=self.controller,
            kinematics=self.kinematics,
            config={
                "dt": dt,
                "max_steps": max_steps,
                "lateral_limit": 0.5,
                "logging_enabled": False,
            },
        )
        result = env.run_episode(seed=0)

        metric_value = result.metrics.get(self.metric, float("inf"))

        trial = {
            "params": params.copy(),
            "metrics": result.metrics.copy(),
            "metric_value": metric_value,
        }
        self._trial_history.append(trial)

        return metric_value

    def tune(
        self,
        trajectory: np.ndarray,
        n_trials: int = 50,
        q_range: tuple[float, float] = (0.1, 10.0),
        r_range: tuple[float, float] = (0.01, 1.0),
        dt: float = 0.02,
        max_steps: int = 1000,
    ) -> TuningResult:
        """Run parameter tuning.

        Parameters
        ----------
        trajectory : ndarray
            Reference trajectory for evaluation
        n_trials : int
            Number of optimization trials
        q_range : tuple
            Range for Q diagonal values
        r_range : tuple
            Range for R diagonal values
        dt : float
            Simulation timestep
        max_steps : int
            Maximum simulation steps

        Returns
        -------
        TuningResult
            Tuning result with best parameters
        """
        self._trial_history = []

        if self.method == "bayesian":
            result = self._tune_bayesian(
                trajectory, n_trials, q_range, r_range, dt, max_steps
            )
        elif self.method == "grid":
            result = self._tune_grid(
                trajectory, n_trials, q_range, r_range, dt, max_steps
            )
        else:
            result = self._tune_random(
                trajectory, n_trials, q_range, r_range, dt, max_steps
            )

        return result

    def _tune_random(
        self,
        trajectory: np.ndarray,
        n_trials: int,
        q_range: tuple[float, float],
        r_range: tuple[float, float],
        dt: float,
        max_steps: int,
    ) -> TuningResult:
        """Random search with log-uniform sampling."""
        best_metric = float("inf")
        best_params = None

        rng = np.random.default_rng()
        log_q = (np.log(q_range[0]), np.log(q_range[1]))
        log_r = (np.log(r_range[0]), np.log(r_range[1]))

        for _ in range(n_trials):
            Q = [float(np.exp(rng.uniform(*log_q))) for _ in range(4)]
            R = [float(np.exp(rng.uniform(*log_r))) for _ in range(2)]

            params = {"Q": Q, "R": R}
            metric = self.objective(params, trajectory, dt, max_steps)

            if metric < best_metric:
                best_metric = metric
                best_params = params.copy()

        return TuningResult(
            best_Q=best_params["Q"],
            best_R=best_params["R"],
            best_metric=best_metric,
            all_trials=self._trial_history,
        )

    def _tune_grid(
        self,
        trajectory: np.ndarray,
        n_trials: int,
        q_range: tuple[float, float],
        r_range: tuple[float, float],
        dt: float,
        max_steps: int,
    ) -> TuningResult:
        """Grid search optimization."""
        n_per_dim = max(2, int(n_trials ** (1 / 6)))

        q_values = np.geomspace(q_range[0], q_range[1], n_per_dim)
        r_values = np.geomspace(r_range[0], r_range[1], n_per_dim)

        best_metric = float("inf")
        best_params = None

        for q_lat in q_values:
            for q_yaw in q_values:
                for r_main in r_values:
                    Q = [float(q_lat), float(q_yaw), float(q_lat * 0.5), float(q_yaw * 0.5)]
                    R = [float(r_main), float(r_main)]

                    params = {"Q": Q, "R": R}
                    metric = self.objective(params, trajectory, dt, max_steps)

                    if metric < best_metric:
                        best_metric = metric
                        best_params = params.copy()

        return TuningResult(
            best_Q=best_params["Q"],
            best_R=best_params["R"],
            best_metric=best_metric,
            all_trials=self._trial_history,
        )

    def _tune_bayesian(
        self,
        trajectory: np.ndarray,
        n_trials: int,
        q_range: tuple[float, float],
        r_range: tuple[float, float],
        dt: float,
        max_steps: int,
    ) -> TuningResult:
        """Bayesian optimization (fallback to random if skopt not available)."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real

            space = [
                Real(q_range[0], q_range[1], name="q_lat", prior="log-uniform"),
                Real(q_range[0], q_range[1], name="q_yaw", prior="log-uniform"),
                Real(q_range[0], q_range[1], name="q_v", prior="log-uniform"),
                Real(q_range[0], q_range[1], name="q_omega", prior="log-uniform"),
                Real(r_range[0], r_range[1], name="r_v", prior="log-uniform"),
                Real(r_range[0], r_range[1], name="r_omega", prior="log-uniform"),
            ]

            def objective_wrapper(x):
                params = {
                    "Q": [x[0], x[1], x[2], x[3]],
                    "R": [x[4], x[5]],
                }
                return self.objective(params, trajectory, dt, max_steps)

            result = gp_minimize(
                objective_wrapper,
                space,
                n_calls=n_trials,
                random_state=42,
            )

            best_params = {
                "Q": [result.x[0], result.x[1], result.x[2], result.x[3]],
                "R": [result.x[4], result.x[5]],
            }

            return TuningResult(
                best_Q=best_params["Q"],
                best_R=best_params["R"],
                best_metric=result.fun,
                all_trials=self._trial_history,
            )

        except ImportError:
            return self._tune_random(
                trajectory, n_trials, q_range, r_range, dt, max_steps
            )

    def get_trial_history(self) -> list[dict]:
        """Get history of all trials."""
        return self._trial_history.copy()
