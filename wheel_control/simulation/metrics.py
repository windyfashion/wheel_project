"""Performance metrics evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics:
    """Performance metrics container."""
    rms_lateral_error: float
    max_lateral_error: float
    rms_yaw_error: float
    max_yaw_error: float
    rms_velocity_error: float
    rms_omega_error: float
    control_smoothness: float
    mean_velocity: float


class MetricEvaluator:
    """Evaluator for tracking performance metrics.
    
    Example
    -------
    >>> evaluator = MetricEvaluator()
    >>> metrics = evaluator.compute_from_steps(step_data)
    >>> print(f"RMS lateral error: {metrics['rms_lateral_error']:.4f}")
    """
    
    def compute(
        self,
        e_lat: np.ndarray,
        e_yaw: np.ndarray,
        e_v: np.ndarray,
        e_omega: np.ndarray,
        v_cmd: np.ndarray | None = None,
        omega_cmd: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute all metrics from error arrays.
        
        Parameters
        ----------
        e_lat, e_yaw, e_v, e_omega : ndarray
            Error time series
        v_cmd, omega_cmd : ndarray, optional
            Control command time series (for smoothness)
        
        Returns
        -------
        dict
            Computed metrics
        """
        metrics = {}
        
        # Lateral error metrics
        metrics["rms_lateral_error"] = float(np.sqrt(np.mean(e_lat ** 2)))
        metrics["max_lateral_error"] = float(np.max(np.abs(e_lat)))
        metrics["mean_lateral_error"] = float(np.mean(np.abs(e_lat)))
        
        # Yaw error metrics
        metrics["rms_yaw_error"] = float(np.sqrt(np.mean(e_yaw ** 2)))
        metrics["max_yaw_error"] = float(np.max(np.abs(e_yaw)))
        metrics["mean_yaw_error"] = float(np.mean(np.abs(e_yaw)))
        
        # Velocity error metrics
        metrics["rms_velocity_error"] = float(np.sqrt(np.mean(e_v ** 2)))
        metrics["mean_velocity_error"] = float(np.mean(np.abs(e_v)))
        
        # Omega error metrics
        metrics["rms_omega_error"] = float(np.sqrt(np.mean(e_omega ** 2)))
        metrics["mean_omega_error"] = float(np.mean(np.abs(e_omega)))
        
        # Control smoothness (jerk measure)
        if v_cmd is not None and len(v_cmd) > 2:
            v_jerk = np.diff(np.diff(v_cmd))
            omega_jerk = np.diff(np.diff(omega_cmd)) if omega_cmd is not None else np.zeros_like(v_jerk)
            metrics["control_smoothness"] = float(
                np.sqrt(np.mean(v_jerk ** 2) + np.mean(omega_jerk ** 2))
            )
        else:
            metrics["control_smoothness"] = 0.0
        
        # Mean velocity (efficiency measure)
        if v_cmd is not None:
            metrics["mean_velocity"] = float(np.mean(np.abs(v_cmd)))
        else:
            metrics["mean_velocity"] = 0.0
        
        return metrics
    
    def compute_from_steps(self, steps: list[dict]) -> dict[str, float]:
        """Compute metrics from step data list.
        
        Parameters
        ----------
        steps : list[dict]
            List of step dictionaries with error keys
        
        Returns
        -------
        dict
            Computed metrics
        """
        if not steps:
            return {
                "rms_lateral_error": 0.0,
                "max_lateral_error": 0.0,
                "rms_yaw_error": 0.0,
                "max_yaw_error": 0.0,
                "rms_velocity_error": 0.0,
                "rms_omega_error": 0.0,
                "control_smoothness": 0.0,
                "mean_velocity": 0.0,
            }
        
        e_lat = np.array([s["e_lat"] for s in steps])
        e_yaw = np.array([s["e_yaw"] for s in steps])
        e_v = np.array([s["e_v"] for s in steps])
        e_omega = np.array([s["e_omega"] for s in steps])
        
        v_cmd = np.array([s.get("v_cmd", 0) for s in steps]) if "v_cmd" in steps[0] else None
        omega_cmd = np.array([s.get("omega_cmd", 0) for s in steps]) if "omega_cmd" in steps[0] else None
        
        return self.compute(e_lat, e_yaw, e_v, e_omega, v_cmd, omega_cmd)
    
    def compute_summary(
        self,
        episodes: list[dict],
    ) -> dict[str, dict[str, float]]:
        """Compute summary statistics across multiple episodes.
        
        Parameters
        ----------
        episodes : list[dict]
            List of episode metric dictionaries
        
        Returns
        -------
        dict
            Summary statistics (mean, std, min, max) for each metric
        """
        if not episodes:
            return {}
        
        summary = {}
        
        # Get all metric keys
        keys = set()
        for ep in episodes:
            keys.update(ep.keys())
        
        for key in keys:
            values = [ep.get(key, 0) for ep in episodes]
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        
        return summary
    
    @staticmethod
    def format_metrics(metrics: dict[str, float], precision: int = 4) -> str:
        """Format metrics for display.
        
        Parameters
        ----------
        metrics : dict
            Metrics dictionary
        precision : int
            Decimal precision
        
        Returns
        -------
        str
            Formatted string
        """
        lines = ["Performance Metrics:"]
        lines.append("-" * 40)
        
        for key, value in sorted(metrics.items()):
            lines.append(f"  {key}: {value:.{precision}f}")
        
        return "\n".join(lines)
