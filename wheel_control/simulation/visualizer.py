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

    Uses ``set_data()`` incremental updates instead of clearing and
    re-drawing every frame, which is significantly faster.

    Example
    -------
    >>> viz = RealtimeVisualizer()
    >>> viz.setup(trajectory)
    >>> for step in range(max_steps):
    ...     state, done, info = env.step()
    ...     viz.update(state, ref_point, errors, dt)
    >>> viz.show()
    """

    def __init__(
        self,
        config: PlotConfig | None = None,
        realtime: bool = True,
        update_interval: int = 5,
    ):
        self.config = config or PlotConfig()
        self.realtime = realtime
        self.update_interval = update_interval

        self._ref_trajectory: np.ndarray | None = None
        self._actual_path: list[np.ndarray] = []
        self._errors: dict[str, list[float]] = {
            "e_lat": [], "e_yaw": [], "e_v": [], "e_omega": [],
            "ref_v": [], "actual_v": [], "kappa": [],
            "time": [],
        }

        self._fig = None
        self._axes = None
        self._lines: dict[str, Any] = {}
        self._initialized = False
        self._step_count = 0

    def setup(self, ref_trajectory: np.ndarray) -> None:
        """Setup visualization with reference trajectory."""
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
        """Setup matplotlib figure with subplots and persistent line objects."""
        import matplotlib.pyplot as plt

        self._fig, self._axes = plt.subplots(
            2, 3,
            figsize=self.config.figsize,
            dpi=self.config.dpi,
        )
        self._fig.suptitle("Trajectory Tracking Visualization", fontsize=14)
        self._axes = self._axes.flatten()

        # --- Subplot 0: trajectory tracking ---
        ax = self._axes[0]
        ax.set_title("Trajectory Tracking")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal")
        ax.grid(self.config.grid)
        ax.plot(
            self._ref_trajectory[:, X],
            self._ref_trajectory[:, Y],
            color=self.config.ref_color, linestyle="--",
            label="Reference", alpha=0.7,
        )
        self._lines["actual_path"], = ax.plot(
            [], [], color=self.config.actual_color,
            label="Actual", linewidth=1.5,
        )
        self._lines["robot_pos"] = ax.scatter(
            [], [], color=self.config.robot_color,
            s=100, marker="o", zorder=5,
        )
        ax.legend(loc="upper right")

        # --- Subplot 1: lateral error ---
        ax = self._axes[1]
        ax.set_title("Lateral Error")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error [m]")
        ax.grid(self.config.grid)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        self._lines["e_lat"], = ax.plot([], [], color=self.config.actual_color)

        # --- Subplot 2: yaw error ---
        ax = self._axes[2]
        ax.set_title("Yaw Error")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error [rad]")
        ax.grid(self.config.grid)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        self._lines["e_yaw"], = ax.plot([], [], color=self.config.actual_color)

        # --- Subplot 3: velocity ---
        ax = self._axes[3]
        ax.set_title("Velocity")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")
        ax.grid(self.config.grid)
        self._lines["ref_v"], = ax.plot(
            [], [], color=self.config.ref_color,
            linestyle="--", label="Reference",
        )
        self._lines["actual_v"], = ax.plot(
            [], [], color=self.config.actual_color, label="Actual",
        )
        ax.legend(loc="upper right")

        # --- Subplot 4: curvature (static) ---
        ax = self._axes[4]
        ax.set_title("Curvature")
        ax.set_xlabel("Arc length [m]")
        ax.set_ylabel("Curvature [1/m]")
        ax.grid(self.config.grid)
        arc = self._compute_arc_length(self._ref_trajectory)
        ax.plot(
            arc, self._ref_trajectory[:, KAPPA],
            color=self.config.ref_color, linestyle="--",
            label="Reference", alpha=0.7,
        )

        # --- Subplot 5: velocity error ---
        ax = self._axes[5]
        ax.set_title("Velocity Error")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error [m/s]")
        ax.grid(self.config.grid)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        self._lines["e_v"], = ax.plot([], [], color=self.config.actual_color)

        plt.tight_layout()
        self._initialized = True

    def update(
        self,
        state: np.ndarray,
        ref_point: np.ndarray,
        errors: dict[str, float],
        dt: float,
    ) -> None:
        """Update visualization with new data."""
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

        if (self.realtime and self._initialized
                and self._step_count % self.update_interval == 0):
            self._update_plots()

    def _update_plots(self) -> None:
        """Incrementally update all plots using set_data."""
        if not self._initialized or self._fig is None:
            return

        import matplotlib.pyplot as plt

        times = self._errors["time"]
        actual = np.array(self._actual_path)

        # Trajectory
        self._lines["actual_path"].set_data(actual[:, 0], actual[:, 1])
        self._lines["robot_pos"].set_offsets(actual[-1:, :2])
        self._axes[0].relim()
        self._axes[0].autoscale_view()

        # Lateral error
        self._lines["e_lat"].set_data(times, self._errors["e_lat"])
        self._axes[1].relim()
        self._axes[1].autoscale_view()

        # Yaw error
        self._lines["e_yaw"].set_data(times, self._errors["e_yaw"])
        self._axes[2].relim()
        self._axes[2].autoscale_view()

        # Velocity
        self._lines["ref_v"].set_data(times, self._errors["ref_v"])
        self._lines["actual_v"].set_data(times, self._errors["actual_v"])
        self._axes[3].relim()
        self._axes[3].autoscale_view()

        # Velocity error
        self._lines["e_v"].set_data(times, self._errors["e_v"])
        self._axes[5].relim()
        self._axes[5].autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def show(self) -> None:
        """Display final visualization."""
        import matplotlib.pyplot as plt

        if not self._initialized:
            self._setup_figure()

        self._update_plots()
        plt.show()

    def save_figure(self, path: str | Path) -> None:
        """Save current figure to file."""
        if not self._initialized:
            self._setup_figure()

        self._update_plots()
        self._fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")

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
        """Plot multiple trajectories for comparison."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal")
        ax.grid(self.config.grid)

        colors = ["blue", "red", "green", "orange", "purple", "cyan"]
        for i, (name, traj) in enumerate(trajectories.items()):
            ax.plot(traj[:, X], traj[:, Y], label=name, color=colors[i % len(colors)])

        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_error_analysis(
        self,
        errors: dict[str, list[float]],
        times: list[float],
    ) -> None:
        """Plot detailed error analysis."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Error Analysis", fontsize=14)
        times_arr = np.array(times)

        for ax, key, title, ylabel in [
            (axes[0, 0], "e_lat",   "Lateral Error",           "Error [m]"),
            (axes[0, 1], "e_yaw",   "Yaw Error",               "Error [rad]"),
            (axes[1, 0], "e_v",     "Velocity Error",          "Error [m/s]"),
            (axes[1, 1], "e_omega", "Angular Velocity Error",  "Error [rad/s]"),
        ]:
            ax.plot(times_arr, errors[key], color="red")
            ax.set_title(title)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.axhline(y=0, color="gray", linestyle="--")

        plt.tight_layout()
        plt.show()
