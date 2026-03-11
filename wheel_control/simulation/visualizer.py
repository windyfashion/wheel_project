"""Real-time visualization for trajectory tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..trajectory.base import X, Y, YAW, VX, W, KAPPA


@dataclass
class PlotConfig:
    """Configuration for visualization plots."""
    figsize: tuple[int, int] = (14, 10)
    dpi: int = 100
    trajectory_color: str = "blue"
    actual_color: str = "red"
    robot_color: str = "green"
    ref_color: str = "gray"
    grid: bool = True
    fontsize: int = 10


class RealtimeVisualizer:
    """Real-time visualization for trajectory tracking.
    
    Provides multiple subplots:
    1. Trajectory tracking (XY plane)
    2. Lateral error curve
    3. Yaw error curve
    4. Velocity curves (ref vs actual)
    5. Curvature curve
    
    Example
    -------
    >>> viz = RealtimeVisualizer()
    >>> for step in range(max_steps):
    ...     state, done, info = env.step()
    ...     viz.update(env)
    >>> viz.show()
    """
    
    def __init__(
        self,
        config: PlotConfig | None = None,
        realtime: bool = True,
        update_interval: int = 5,
    ):
        """Initialize visualizer.
        
        Parameters
        ----------
        config : PlotConfig, optional
            Plot configuration
        realtime : bool
            Enable real-time updates
        update_interval : int
            Update every N steps
        """
        self.config = config or PlotConfig()
        self.realtime = realtime
        self.update_interval = update_interval
        
        # Data storage
        self._ref_trajectory: np.ndarray | None = None
        self._actual_path: list[np.ndarray] = []
        self._errors: dict[str, list[float]] = {
            "e_lat": [], "e_yaw": [], "e_v": [], "e_omega": [],
            "ref_v": [], "actual_v": [], "kappa": [],
            "time": [],
        }
        
        # Plot objects
        self._fig = None
        self._axes = None
        self._initialized = False
        self._step_count = 0
    
    def setup(self, ref_trajectory: np.ndarray) -> None:
        """Setup visualization with reference trajectory.
        
        Parameters
        ----------
        ref_trajectory : ndarray, shape (N, 7)
            Reference trajectory
        """
        self._ref_trajectory = ref_trajectory
        self._actual_path = []
        self._errors = {
            "e_lat": [], "e_yaw": [], "e_v": [], "e_omega": [],
            "ref_v": [], "actual_v": [], "kappa": [],
            "time": [],
        }
        self._step_count = 0
        self._initialized = False
        
        if self.realtime:
            self._setup_figure()
    
    def _setup_figure(self) -> None:
        """Setup matplotlib figure with subplots."""
        import matplotlib.pyplot as plt
        
        self._fig, self._axes = plt.subplots(
            2, 3,
            figsize=self.config.figsize,
            dpi=self.config.dpi,
        )
        self._fig.suptitle("Trajectory Tracking Visualization", fontsize=14)
        
        # Flatten axes for easier indexing
        self._axes = self._axes.flatten()
        
        # Trajectory plot (larger)
        ax_traj = self._axes[0]
        ax_traj.set_title("Trajectory Tracking")
        ax_traj.set_xlabel("X [m]")
        ax_traj.set_ylabel("Y [m]")
        ax_traj.set_aspect("equal")
        ax_traj.grid(self.config.grid)
        
        # Plot reference trajectory
        ax_traj.plot(
            self._ref_trajectory[:, X],
            self._ref_trajectory[:, Y],
            color=self.config.ref_color,
            linestyle="--",
            label="Reference",
            alpha=0.7,
        )
        
        # Lateral error
        ax_elat = self._axes[1]
        ax_elat.set_title("Lateral Error")
        ax_elat.set_xlabel("Time [s]")
        ax_elat.set_ylabel("Error [m]")
        ax_elat.grid(self.config.grid)
        
        # Yaw error
        ax_eyaw = self._axes[2]
        ax_eyaw.set_title("Yaw Error")
        ax_eyaw.set_xlabel("Time [s]")
        ax_eyaw.set_ylabel("Error [rad]")
        ax_eyaw.grid(self.config.grid)
        
        # Velocity
        ax_vel = self._axes[3]
        ax_vel.set_title("Velocity")
        ax_vel.set_xlabel("Time [s]")
        ax_vel.set_ylabel("Velocity [m/s]")
        ax_vel.grid(self.config.grid)
        
        # Curvature
        ax_kappa = self._axes[4]
        ax_kappa.set_title("Curvature")
        ax_kappa.set_xlabel("Arc length [m]")
        ax_kappa.set_ylabel("Curvature [1/m]")
        ax_kappa.grid(self.config.grid)
        
        # Plot reference curvature
        arc = self._compute_arc_length(self._ref_trajectory)
        ax_kappa.plot(
            arc,
            self._ref_trajectory[:, KAPPA],
            color=self.config.ref_color,
            linestyle="--",
            label="Reference",
            alpha=0.7,
        )
        
        # Control commands
        ax_ctrl = self._axes[5]
        ax_ctrl.set_title("Control Commands")
        ax_ctrl.set_xlabel("Time [s]")
        ax_ctrl.set_ylabel("Command")
        ax_ctrl.grid(self.config.grid)
        
        plt.tight_layout()
        self._initialized = True
    
    def update(
        self,
        state: np.ndarray,
        ref_point: np.ndarray,
        errors: dict[str, float],
        dt: float,
    ) -> None:
        """Update visualization with new data.
        
        Parameters
        ----------
        state : ndarray
            Current robot state [x, y, theta, vx, omega]
        ref_point : ndarray
            Current reference point
        errors : dict
            Tracking errors
        dt : float
            Time step
        """
        # Store data
        self._actual_path.append(state[:3].copy())
        
        time = self._step_count * dt
        self._errors["time"].append(time)
        self._errors["e_lat"].append(errors.get("e_lat", 0))
        self._errors["e_yaw"].append(errors.get("e_yaw", 0))
        self._errors["e_v"].append(errors.get("e_v", 0))
        self._errors["e_omega"].append(errors.get("e_omega", 0))
        self._errors["ref_v"].append(ref_point[VX])
        self._errors["actual_v"].append(state[3])
        self._errors["kappa"].append(ref_point[KAPPA])
        
        self._step_count += 1
        
        # Update plots
        if self.realtime and self._initialized and self._step_count % self.update_interval == 0:
            self._update_plots()
    
    def _update_plots(self) -> None:
        """Update all plots with current data."""
        if not self._initialized or self._fig is None:
            return
        
        import matplotlib.pyplot as plt
        
        times = np.array(self._errors["time"])
        
        # Update trajectory plot
        ax_traj = self._axes[0]
        # Clear only the actual path line
        for line in ax_traj.lines[1:]:  # Keep reference line
            line.remove()
        
        actual = np.array(self._actual_path)
        ax_traj.plot(
            actual[:, 0], actual[:, 1],
            color=self.config.actual_color,
            label="Actual",
            linewidth=1.5,
        )
        
        # Plot robot position
        if len(actual) > 0:
            ax_traj.scatter(
                actual[-1, 0], actual[-1, 1],
                color=self.config.robot_color,
                s=100,
                marker="o",
                zorder=5,
            )
        
        ax_traj.legend(loc="upper right")
        
        # Update error plots
        ax_elat = self._axes[1]
        ax_elat.clear()
        ax_elat.set_title("Lateral Error")
        ax_elat.set_xlabel("Time [s]")
        ax_elat.set_ylabel("Error [m]")
        ax_elat.grid(self.config.grid)
        ax_elat.plot(times, self._errors["e_lat"], color=self.config.actual_color)
        ax_elat.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        ax_eyaw = self._axes[2]
        ax_eyaw.clear()
        ax_eyaw.set_title("Yaw Error")
        ax_eyaw.set_xlabel("Time [s]")
        ax_eyaw.set_ylabel("Error [rad]")
        ax_eyaw.grid(self.config.grid)
        ax_eyaw.plot(times, self._errors["e_yaw"], color=self.config.actual_color)
        ax_eyaw.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        # Update velocity plot
        ax_vel = self._axes[3]
        ax_vel.clear()
        ax_vel.set_title("Velocity")
        ax_vel.set_xlabel("Time [s]")
        ax_vel.set_ylabel("Velocity [m/s]")
        ax_vel.grid(self.config.grid)
        ax_vel.plot(times, self._errors["ref_v"], color=self.config.ref_color, 
                   linestyle="--", label="Reference")
        ax_vel.plot(times, self._errors["actual_v"], color=self.config.actual_color,
                   label="Actual")
        ax_vel.legend(loc="upper right")
        
        # Curvature plot stays static (reference only)
        # Control commands
        ax_ctrl = self._axes[5]
        ax_ctrl.clear()
        ax_ctrl.set_title("Velocity Error")
        ax_ctrl.set_xlabel("Time [s]")
        ax_ctrl.set_ylabel("Error [m/s]")
        ax_ctrl.grid(self.config.grid)
        ax_ctrl.plot(times, self._errors["e_v"], color=self.config.actual_color)
        ax_ctrl.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plt.pause(0.001)
    
    def show(self) -> None:
        """Display final visualization."""
        import matplotlib.pyplot as plt
        
        if not self._initialized:
            self._setup_figure()
        
        self._update_plots()
        plt.show()
    
    def save_figure(self, path: str | Path) -> None:
        """Save current figure to file.
        
        Parameters
        ----------
        path : str or Path
            Output file path
        """
        import matplotlib.pyplot as plt
        
        if not self._initialized:
            self._setup_figure()
        
        self._update_plots()
        self._fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
    
    def save_animation(
        self,
        path: str | Path,
        fps: int = 30,
    ) -> None:
        """Save animation to file.
        
        Requires matplotlib animation support.
        
        Parameters
        ----------
        path : str or Path
            Output file path (.mp4 or .gif)
        fps : int
            Frames per second
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, PillowWriter
        
        if not self._initialized:
            self._setup_figure()
        
        # Determine writer based on file extension
        path = Path(path)
        if path.suffix == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps)
        
        with writer.saving(self._fig, str(path), dpi=self.config.dpi):
            # Write final frame
            self._update_plots()
            writer.grab_frame()
    
    def _compute_arc_length(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute arc length along trajectory."""
        diffs = np.diff(trajectory[:, :2], axis=0)
        seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.concatenate([[0], np.cumsum(seg_lengths)])
    
    def plot_comparison(
        self,
        trajectories: dict[str, np.ndarray],
        title: str = "Trajectory Comparison",
    ) -> None:
        """Plot multiple trajectories for comparison.
        
        Parameters
        ----------
        trajectories : dict
            Dictionary of name -> trajectory arrays
        title : str
            Plot title
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal")
        ax.grid(self.config.grid)
        
        colors = ["blue", "red", "green", "orange", "purple", "cyan"]
        
        for i, (name, traj) in enumerate(trajectories.items()):
            color = colors[i % len(colors)]
            ax.plot(traj[:, X], traj[:, Y], label=name, color=color)
        
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_error_analysis(
        self,
        errors: dict[str, list[float]],
        times: list[float],
    ) -> None:
        """Plot detailed error analysis.
        
        Parameters
        ----------
        errors : dict
            Error time series
        times : list
            Time points
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Error Analysis", fontsize=14)
        
        times = np.array(times)
        
        # Lateral error
        axes[0, 0].plot(times, errors["e_lat"], color="red")
        axes[0, 0].set_title("Lateral Error")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("Error [m]")
        axes[0, 0].grid(True)
        axes[0, 0].axhline(y=0, color="gray", linestyle="--")
        
        # Yaw error
        axes[0, 1].plot(times, errors["e_yaw"], color="red")
        axes[0, 1].set_title("Yaw Error")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("Error [rad]")
        axes[0, 1].grid(True)
        axes[0, 1].axhline(y=0, color="gray", linestyle="--")
        
        # Velocity error
        axes[1, 0].plot(times, errors["e_v"], color="red")
        axes[1, 0].set_title("Velocity Error")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("Error [m/s]")
        axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0, color="gray", linestyle="--")
        
        # Omega error
        axes[1, 1].plot(times, errors["e_omega"], color="red")
        axes[1, 1].set_title("Angular Velocity Error")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("Error [rad/s]")
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=0, color="gray", linestyle="--")
        
        plt.tight_layout()
        plt.show()
