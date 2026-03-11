"""Bezier curve path planner."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


class BezierPathPlanner:
    """Path planner using Bezier curves.
    
    Generates smooth paths between waypoints using quintic Bezier curves
    for better curvature continuity.
    """
    
    def __init__(
        self,
        n_points: int = 100,
        smoothness: float = 0.5,
    ):
        """Initialize Bezier path planner.
        
        Parameters
        ----------
        n_points : int
            Number of points in output path
        smoothness : float
            Control point offset factor (0-1)
        """
        self.n_points = n_points
        self.smoothness = smoothness
    
    def plan(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Plan a Bezier path from start to goal.
        
        For simple start-goal, generates a smooth connecting curve.
        For waypoints, generates a spline through all points.
        
        Parameters
        ----------
        start : ndarray
            Start position [x, y] or waypoints [(x, y), ...]
        goal : ndarray
            Goal position [x, y] (ignored if start is waypoints)
        
        Returns
        -------
        ndarray, shape (N, 2)
            Path waypoints
        """
        start = np.atleast_2d(start)
        goal = np.atleast_2d(goal)
        
        # If start has multiple points, treat as waypoints
        if len(start) > 1:
            waypoints = start
        else:
            waypoints = np.vstack([start, goal])
        
        return self._generate_bezier_path(waypoints)
    
    def _generate_bezier_path(self, waypoints: np.ndarray) -> np.ndarray:
        """Generate smooth path through waypoints.
        
        Parameters
        ----------
        waypoints : ndarray, shape (M, 2)
            Waypoint coordinates
        
        Returns
        -------
        ndarray, shape (N, 2)
            Smooth path
        """
        if len(waypoints) < 2:
            return waypoints
        
        # Compute arc length parameter
        diffs = np.diff(waypoints, axis=0)
        seg_lengths = np.sqrt((diffs**2).sum(axis=1))
        arc = np.concatenate([[0], np.cumsum(seg_lengths)])
        
        # Fit cubic spline
        cs_x = CubicSpline(arc, waypoints[:, 0])
        cs_y = CubicSpline(arc, waypoints[:, 1])
        
        # Sample uniformly in arc length
        s = np.linspace(0, arc[-1], self.n_points)
        x = cs_x(s)
        y = cs_y(s)
        
        return np.column_stack([x, y])
    
    def plan_with_heading(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        start_heading: float | None = None,
        goal_heading: float | None = None,
    ) -> np.ndarray:
        """Plan path with specified start and goal headings.
        
        Uses quintic Bezier curve for smooth curvature.
        
        Parameters
        ----------
        start : ndarray, shape (2,)
            Start position
        goal : ndarray, shape (2,)
            Goal position
        start_heading : float, optional
            Start heading in radians
        goal_heading : float, optional
            Goal heading in radians
        
        Returns
        -------
        ndarray, shape (N, 2)
            Path waypoints
        """
        start = np.asarray(start)
        goal = np.asarray(goal)
        
        # Compute headings if not provided
        if start_heading is None:
            start_heading = np.arctan2(goal[1] - start[1], goal[0] - start[0])
        if goal_heading is None:
            goal_heading = np.arctan2(goal[1] - start[1], goal[0] - start[0])
        
        # Distance for control point placement
        dist = np.linalg.norm(goal - start)
        offset = dist * self.smoothness * 0.5
        
        # Compute control points for quintic Bezier
        start_dir = np.array([np.cos(start_heading), np.sin(start_heading)])
        goal_dir = np.array([np.cos(goal_heading), np.sin(goal_heading)])
        
        # P0, P1, P2 along start tangent
        p0 = start
        p1 = start + offset * start_dir
        p2 = start + 2 * offset * start_dir
        
        # P3, P4, P5 along goal tangent
        p5 = goal
        p4 = goal - offset * goal_dir
        p3 = goal - 2 * offset * goal_dir
        
        control_points = np.array([p0, p1, p2, p3, p4, p5])
        
        return self._evaluate_quintic_bezier(control_points)
    
    def _evaluate_quintic_bezier(self, control_points: np.ndarray) -> np.ndarray:
        """Evaluate quintic Bezier curve.
        
        Parameters
        ----------
        control_points : ndarray, shape (6, 2)
            Control points P0-P5
        
        Returns
        -------
        ndarray, shape (N, 2)
            Curve points
        """
        coeffs = np.array([1, 5, 10, 10, 5, 1], dtype=float)
        t = np.linspace(0, 1, self.n_points)[:, np.newaxis]
        
        # Bernstein basis polynomials
        basis = np.zeros((self.n_points, 6))
        for i in range(6):
            basis[:, i] = coeffs[i] * (1 - t[:, 0]) ** (5 - i) * t[:, 0] ** i
        
        return basis @ control_points
