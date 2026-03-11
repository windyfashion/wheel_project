"""Control logger for recording simulation data."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class StepData:
    """Data recorded at each simulation step."""
    step: int
    time: float
    
    # Robot state
    x: float
    y: float
    theta: float
    vx: float
    omega: float
    
    # Reference state
    ref_x: float
    ref_y: float
    ref_yaw: float
    ref_vx: float
    ref_omega: float
    ref_kappa: float
    
    # Frenet errors
    e_lat: float
    e_yaw: float
    e_v: float
    e_omega: float
    
    # Control commands
    v_cmd: float
    omega_cmd: float


class ControlLogger:
    """Logger for control process data.
    
    Records step-by-step data and provides export functionality.
    
    Example
    -------
    >>> logger = ControlLogger("logs")
    >>> logger.start_episode()
    >>> logger.log_step(step_data)
    >>> logger.end_episode()
    >>> logger.export_csv("episode_001.csv")
    """
    
    def __init__(
        self,
        log_dir: str | Path = "logs",
        level: int = logging.INFO,
    ):
        """Initialize logger.
        
        Parameters
        ----------
        log_dir : str or Path
            Directory for log files
        level : int
            Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = logging.getLogger("wheel_control")
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Episode data storage
        self._current_episode: list[StepData] = []
        self._episodes: list[list[StepData]] = []
        self._episode_start_time: datetime | None = None
    
    def start_episode(self) -> None:
        """Start a new episode recording."""
        self._current_episode = []
        self._episode_start_time = datetime.now()
        self.logger.info("Episode started")
    
    def log_step(self, data: StepData) -> None:
        """Log data for a single step.
        
        Parameters
        ----------
        data : StepData
            Step data to record
        """
        self._current_episode.append(data)
    
    def log_step_dict(self, **kwargs: Any) -> None:
        """Log step data from keyword arguments.
        
        Parameters
        ----------
        **kwargs : dict
            Step data fields
        """
        data = StepData(**kwargs)
        self.log_step(data)
    
    def end_episode(self) -> str:
        """End current episode and save.
        
        Returns
        -------
        str
            Episode identifier
        """
        if not self._current_episode:
            self.logger.warning("Attempted to end empty episode")
            return ""
        
        episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episodes.append(self._current_episode)
        
        n_steps = len(self._current_episode)
        self.logger.info(f"Episode ended: {n_steps} steps recorded")
        
        self._current_episode = []
        return episode_id
    
    def get_current_episode(self) -> list[StepData]:
        """Get current episode data.
        
        Returns
        -------
        list[StepData]
            List of step data for current episode
        """
        return self._current_episode.copy()
    
    def export_csv(self, path: str | Path | None = None) -> Path:
        """Export current episode data to CSV.
        
        Parameters
        ----------
        path : str or Path, optional
            Output file path. If None, generates automatic name.
        
        Returns
        -------
        Path
            Path to exported file
        """
        if not self._current_episode:
            raise ValueError("No episode data to export")
        
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.log_dir / f"episode_{timestamp}.csv"
        else:
            path = Path(path)
        
        fieldnames = [
            "step", "time",
            "x", "y", "theta", "vx", "omega",
            "ref_x", "ref_y", "ref_yaw", "ref_vx", "ref_omega", "ref_kappa",
            "e_lat", "e_yaw", "e_v", "e_omega",
            "v_cmd", "omega_cmd",
        ]
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in self._current_episode:
                row = {field: getattr(data, field) for field in fieldnames}
                writer.writerow(row)
        
        self.logger.info(f"Episode data exported to {path}")
        return path
    
    def export_summary(self, path: str | Path | None = None) -> Path:
        """Export summary statistics for all episodes.
        
        Parameters
        ----------
        path : str or Path, optional
            Output file path
        
        Returns
        -------
        Path
            Path to exported file
        """
        if path is None:
            path = self.log_dir / "summary.csv"
        else:
            path = Path(path)
        
        import numpy as np
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "n_steps",
                "rms_e_lat", "max_e_lat",
                "rms_e_yaw", "max_e_yaw",
                "rms_e_v", "rms_e_omega",
            ])
            
            for i, episode in enumerate(self._episodes):
                if not episode:
                    continue
                
                e_lats = [d.e_lat for d in episode]
                e_yaws = [d.e_yaw for d in episode]
                e_vs = [d.e_v for d in episode]
                e_omegas = [d.e_omega for d in episode]
                
                writer.writerow([
                    i, len(episode),
                    np.sqrt(np.mean(np.array(e_lats)**2)),
                    np.max(np.abs(e_lats)),
                    np.sqrt(np.mean(np.array(e_yaws)**2)),
                    np.max(np.abs(e_yaws)),
                    np.sqrt(np.mean(np.array(e_vs)**2)),
                    np.sqrt(np.mean(np.array(e_omegas)**2)),
                ])
        
        self.logger.info(f"Summary exported to {path}")
        return path
    
    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)
    
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)
