"""LQR parameter auto-tuning module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .lqr import LQRController
from ..kinematics import DiffDriveKinematics
from ..simulation.metrics import MetricEvaluator


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
    >>> tuner = LQRTuner(controller, kinematics, method="bayesian")
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
        """Initialize tuner.
        
        Parameters
        ----------
        controller : LQRController
            LQR controller to tune
        kinematics : DiffDriveKinematics
            Robot kinematics model
        method : str
            Optimization method: "bayesian", "grid", or "random"
        metric : str
            Metric to optimize (minimize)
        """
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
        # Set new weights
        self.controller.set_weights(params["Q"], params["R"])
        self.controller.reset()
        
        # Run simulation
        episode_data = self._run_episode(trajectory, dt, max_steps)
        
        # Compute metrics
        evaluator = MetricEvaluator()
        metrics = evaluator.compute_from_steps(episode_data)
        
        # Store trial
        trial = {
            "params": params.copy(),
            "metrics": metrics.copy(),
            "metric_value": metrics[self.metric],
        }
        self._trial_history.append(trial)
        
        return metrics[self.metric]
    
    def _run_episode(
        self,
        trajectory: np.ndarray,
        dt: float,
        max_steps: int,
    ) -> list[dict]:
        """Run a single tracking episode.
        
        Parameters
        ----------
        trajectory : ndarray
            Reference trajectory
        dt : float
            Timestep
        max_steps : int
            Maximum steps
        
        Returns
        -------
        list[dict]
            Episode step data
        """
        from ..utils.frenet import FrenetFrame
        from ..trajectory.base import X, Y, YAW, VX, W, KAPPA
        
        # Reset
        self.kinematics.reset(
            x=trajectory[0, X],
            y=trajectory[0, Y],
            theta=trajectory[0, YAW],
            vx=0.0,
            omega=0.0,
        )
        self.controller.reset()
        
        steps_data = []
        nearest_idx = 0
        
        for step in range(max_steps):
            state = self.kinematics.state
            
            # Find nearest point
            nearest_idx = FrenetFrame.find_nearest_point_with_progress(
                state[0], state[1], trajectory, nearest_idx
            )
            
            # Get control
            output = self.controller.compute_control(state, trajectory, nearest_idx)
            
            # Step kinematics
            self.kinematics.step(output.v_cmd, output.omega_cmd)
            
            # Compute errors for logging
            ref = trajectory[nearest_idx]
            frenet = FrenetFrame.world_to_frenet(
                state[0], state[1], state[2], state[3], state[4], ref
            )
            
            steps_data.append({
                "e_lat": frenet.e_lat,
                "e_yaw": frenet.e_yaw,
                "e_v": frenet.e_v,
                "e_omega": frenet.e_omega,
            })
            
            # Check if done
            if nearest_idx >= len(trajectory) - 5:
                break
        
        return steps_data
    
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
        """Random search optimization."""
        best_metric = float("inf")
        best_params = None
        
        rng = np.random.default_rng()
        
        for _ in range(n_trials):
            # Sample random parameters (log-uniform for better coverage)
            Q = [rng.uniform(q_range[0], q_range[1]) for _ in range(4)]
            R = [rng.uniform(r_range[0], r_range[1]) for _ in range(2)]
            
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
        # Determine grid resolution
        n_per_dim = max(2, int(n_trials ** (1/6)))  # 6 parameters total
        
        q_values = np.linspace(q_range[0], q_range[1], n_per_dim)
        r_values = np.linspace(r_range[0], r_range[1], n_per_dim)
        
        best_metric = float("inf")
        best_params = None
        
        # Simplified grid: same Q for e_lat and e_yaw, same R for both
        for q_main in q_values:
            for q_vel in q_values:
                for r_main in r_values:
                    Q = [q_main, q_main, q_vel, q_vel]  # [e_lat, e_yaw, e_v, e_omega]
                    R = [r_main, r_main]
                    
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
            
            # Define search space
            space = [
                Real(q_range[0], q_range[1], name="q_lat"),
                Real(q_range[0], q_range[1], name="q_yaw"),
                Real(q_range[0], q_range[1], name="q_v"),
                Real(q_range[0], q_range[1], name="q_omega"),
                Real(r_range[0], r_range[1], name="r_v"),
                Real(r_range[0], r_range[1], name="r_omega"),
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
            # Fallback to random search
            return self._tune_random(
                trajectory, n_trials, q_range, r_range, dt, max_steps
            )
    
    def get_trial_history(self) -> list[dict]:
        """Get history of all trials.
        
        Returns
        -------
        list[dict]
            List of trial dictionaries
        """
        return self._trial_history.copy()
