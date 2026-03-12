"""Built-in trajectory generators for training."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .base import TrajectoryBase


class CircleTrajectory(TrajectoryBase):
    def __init__(
        self,
        dt: float = 0.02,
        max_v: float = 1.5,
        max_omega: float = 3.0,
        max_acc: float = 1.0,
        n_points: int = 600,
    ):
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_acc = max_acc
        self.n_points = n_points

    def generate(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        radius = rng.uniform(1.5, 5.0)

        t = np.linspace(0, 2.0 * np.pi, self.n_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)

        yaw, vx, omega, kappa = self.compute_derivatives(
            x, y, self.dt, self.max_v, self.max_omega, self.max_acc
        )
        vy = np.zeros(self.n_points)

        return np.stack([x, y, yaw, vx, vy, omega, kappa], axis=1)


class SineTrajectory(TrajectoryBase):
    def __init__(
        self,
        dt: float = 0.02,
        max_v: float = 1.5,
        max_omega: float = 3.0,
        max_acc: float = 1.0,
        n_points: int = 600,
    ):
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_acc = max_acc
        self.n_points = n_points

    def generate(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        amplitude = rng.uniform(1.0, 3.0)
        freq = rng.uniform(0.3, 1.0)
        length = rng.uniform(8.0, 15.0)

        s = np.linspace(0, length, self.n_points)
        x = s
        y = amplitude * np.sin(2.0 * np.pi * freq * s / length)

        yaw, vx, omega, kappa = self.compute_derivatives(
            x, y, self.dt, self.max_v, self.max_omega, self.max_acc
        )
        vy = np.zeros(self.n_points)

        return np.stack([x, y, yaw, vx, vy, omega, kappa], axis=1)


class FigureEightTrajectory(TrajectoryBase):
    def __init__(
        self,
        dt: float = 0.02,
        max_v: float = 1.5,
        max_omega: float = 3.0,
        max_acc: float = 1.0,
        n_points: int = 600,
    ):
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_acc = max_acc
        self.n_points = n_points

    def generate(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        a = rng.uniform(2.0, 4.5)

        t = np.linspace(0, 2.0 * np.pi, self.n_points)
        denom = 1.0 + np.sin(t) ** 2
        x = a * np.cos(t) / denom
        y = a * np.sin(t) * np.cos(t) / denom

        yaw, vx, omega, kappa = self.compute_derivatives(
            x, y, self.dt, self.max_v, self.max_omega, self.max_acc
        )
        vy = np.zeros(self.n_points)

        return np.stack([x, y, yaw, vx, vy, omega, kappa], axis=1)


class StraightTrajectory(TrajectoryBase):
    def __init__(
        self,
        dt: float = 0.02,
        max_v: float = 1.5,
        max_omega: float = 3.0,
        max_acc: float = 1.0,
        n_points: int = 600,
    ):
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_acc = max_acc
        self.n_points = n_points

    def generate(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        heading = rng.uniform(-np.pi, np.pi)
        length = rng.uniform(5.0, 15.0)

        s = np.linspace(0, length, self.n_points)
        x = s * np.cos(heading)
        y = s * np.sin(heading)

        yaw, vx, omega, kappa = self.compute_derivatives(
            x, y, self.dt, self.max_v, self.max_omega, self.max_acc
        )
        vy = np.zeros(self.n_points)

        return np.stack([x, y, yaw, vx, vy, omega, kappa], axis=1)


class BezierSplineTrajectory(TrajectoryBase):
    """Short-range trajectory using quintic Bezier + cubic spline.

    Generates smooth paths of 0.5-3 m with random start-pose lateral / heading
    offsets.  The Bezier curve is sampled with curvature-adaptive density so that
    high-curvature regions get more key-points before cubic-spline fitting.
    """

    _QUINTIC_COEFFS = np.array([1, 5, 10, 10, 5, 1], dtype=float)

    def __init__(
        self,
        dt: float = 0.02,
        max_v: float = 1.5,
        max_omega: float = 3.0,
        max_acc: float = 1.0,
        n_points: int = 600,
        dist_range: tuple[float, float] = (0.5, 3.0),
        lateral_error_range: tuple[float, float] = (0.0, 0.2),
        heading_error_deg_range: tuple[float, float] = (0.0, 20.0),
    ):
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_acc = max_acc
        self.n_points = n_points
        self.dist_range = dist_range
        self.lateral_error_range = lateral_error_range
        self.heading_error_range = (
            np.deg2rad(heading_error_deg_range[0]),
            np.deg2rad(heading_error_deg_range[1]),
        )

    # ------------------------------------------------------------------
    # Quintic Bezier helpers
    # ------------------------------------------------------------------

    @classmethod
    def _quintic_bezier(cls, t: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        """Evaluate quintic Bezier.  t: (M,), ctrl: (6, 2) -> (M, 2)."""
        t = np.asarray(t, dtype=float)[:, None]
        powers = np.arange(6)
        basis = cls._QUINTIC_COEFFS * (1.0 - t) ** (5 - powers) * t ** powers
        return basis @ ctrl

    @staticmethod
    def _menger_curvature(pts: np.ndarray) -> np.ndarray:
        """Vectorised Menger curvature for consecutive point triplets.

        pts: (N, 2) -> kappa: (N,)
        """
        p0, p1, p2 = pts[:-2], pts[1:-1], pts[2:]
        cross = (
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        )
        d01 = np.linalg.norm(p1 - p0, axis=1)
        d12 = np.linalg.norm(p2 - p1, axis=1)
        d02 = np.linalg.norm(p2 - p0, axis=1)
        kappa_mid = 2.0 * np.abs(cross) / (d01 * d12 * d02 + 1e-12)
        kappa = np.empty(len(pts))
        kappa[0] = kappa_mid[0]
        kappa[1:-1] = kappa_mid
        kappa[-1] = kappa_mid[-1]
        return kappa

    def _adaptive_sample(
        self,
        ctrl: np.ndarray,
        n_key: int = 40,
        kappa_weight: float = 8.0,
    ) -> np.ndarray:
        """Curvature-adaptive Bezier sampling -> (n_key, 2) key waypoints."""
        t_dense = np.linspace(0, 1, 500)
        pts_dense = self._quintic_bezier(t_dense, ctrl)
        kappa = self._menger_curvature(pts_dense)

        weights = 1.0 + kappa_weight * kappa
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]

        quantiles = np.linspace(0, 1, n_key)
        t_adaptive = np.interp(quantiles, cdf, t_dense)
        t_adaptive[0], t_adaptive[-1] = 0.0, 1.0

        return self._quintic_bezier(t_adaptive, ctrl)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def generate(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()

        dist = rng.uniform(*self.dist_range)
        lat_err = rng.uniform(*self.lateral_error_range) * rng.choice([-1, 1])
        head_err = rng.uniform(*self.heading_error_range) * rng.choice([-1, 1])

        start_pos = np.array([0.0, lat_err])
        end_pos = np.array([dist, 0.0])
        start_dir = np.array([np.cos(head_err), np.sin(head_err)])
        end_dir = np.array([1.0, 0.0])

        # P0-P1-P2 collinear along start tangent,
        # P3-P4-P5 collinear along end tangent -> near-zero curvature at ends
        step = dist / 5.0
        ctrl = np.array([
            start_pos,
            start_pos + step * start_dir,
            start_pos + 2.0 * step * start_dir,
            end_pos - 2.0 * step * end_dir,
            end_pos - step * end_dir,
            end_pos,
        ])

        key_pts = self._adaptive_sample(ctrl)

        # Arc-length parameterised cubic spline
        diffs = np.diff(key_pts, axis=0)
        seg_len = np.sqrt((diffs**2).sum(axis=1))
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])

        cs_x = CubicSpline(arc, key_pts[:, 0])
        cs_y = CubicSpline(arc, key_pts[:, 1])

        s = np.linspace(0, arc[-1], self.n_points)
        x = cs_x(s)
        y = cs_y(s)

        # Random global rotation so paths aren't always axis-aligned
        base_angle = rng.uniform(-np.pi, np.pi)
        cos_a, sin_a = np.cos(base_angle), np.sin(base_angle)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        yaw, vx, omega, kappa = self.compute_derivatives(
            x_rot, y_rot, self.dt, self.max_v, self.max_omega, self.max_acc
        )
        vy = np.zeros(self.n_points)

        return np.stack([x_rot, y_rot, yaw, vx, vy, omega, kappa], axis=1)


# Registry for lookup by name
_GENERATORS: dict[str, type[TrajectoryBase]] = {
    "circle": CircleTrajectory,
    "sine": SineTrajectory,
    "figure8": FigureEightTrajectory,
    "straight": StraightTrajectory,
    "bezier_spline": BezierSplineTrajectory,
}


class RandomTrajectoryGenerator:
    """Randomly selects and generates trajectories from the built-in set."""

    def __init__(
        self,
        types: list[str] | None = None,
        dt: float = 0.02,
        max_v: float = 1.5,
        max_omega: float = 3.0,
        max_acc: float = 1.0,
        n_points: int = 600,
    ):
        types = types or list(_GENERATORS.keys())
        self.generators = []
        for name in types:
            cls = _GENERATORS[name]
            self.generators.append(
                cls(dt=dt, max_v=max_v, max_omega=max_omega,
                    max_acc=max_acc, n_points=n_points)
            )

    def generate(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        gen = rng.choice(self.generators)
        return gen.generate(rng)
