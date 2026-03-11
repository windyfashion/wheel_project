"""Unit tests for Frenet frame utilities."""

import numpy as np
import pytest

from wheel_control.utils import FrenetFrame, FrenetState


class TestFrenetFrame:
    """Tests for FrenetFrame coordinate transformations."""
    
    def test_world_to_frenet_zero_error(self):
        """Test transformation when robot is on reference."""
        ref_point = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        
        state = FrenetFrame.world_to_frenet(
            x=0.0, y=0.0, theta=0.0, vx=0.5, omega=0.0,
            ref_point=ref_point,
        )
        
        assert np.isclose(state.e_lat, 0.0, atol=1e-6)
        assert np.isclose(state.e_yaw, 0.0, atol=1e-6)
        assert np.isclose(state.e_v, 0.0, atol=1e-6)
    
    def test_world_to_frenet_lateral_error(self):
        """Test lateral error computation."""
        ref_point = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        
        # Robot 0.1m to the left
        state = FrenetFrame.world_to_frenet(
            x=0.0, y=0.1, theta=0.0, vx=0.5, omega=0.0,
            ref_point=ref_point,
        )
        
        assert np.isclose(state.e_lat, 0.1, atol=1e-6)
    
    def test_world_to_frenet_yaw_error(self):
        """Test yaw error computation."""
        ref_point = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        
        # Robot yawed 0.1 rad
        state = FrenetFrame.world_to_frenet(
            x=0.0, y=0.0, theta=0.1, vx=0.5, omega=0.0,
            ref_point=ref_point,
        )
        
        assert np.isclose(state.e_yaw, 0.1, atol=1e-6)
    
    def test_frenet_to_world_roundtrip(self):
        """Test roundtrip transformation."""
        ref_point = np.array([1.0, 2.0, 0.5, 0.5, 0.0, 0.1, 0.1])
        
        # Forward transform
        frenet = FrenetFrame.world_to_frenet(
            x=1.1, y=2.1, theta=0.6, vx=0.4, omega=0.1,
            ref_point=ref_point,
        )
        
        # Backward transform
        x, y, theta = FrenetFrame.frenet_to_world(
            e_s=frenet.e_s,
            e_lat=frenet.e_lat,
            e_yaw=frenet.e_yaw,
            ref_point=ref_point,
        )
        
        assert np.isclose(x, 1.1, atol=1e-6)
        assert np.isclose(y, 2.1, atol=1e-6)
        assert np.isclose(theta, 0.6, atol=1e-6)
    
    def test_find_nearest_point(self):
        """Test finding nearest point on trajectory."""
        trajectory = np.array([
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        ])
        
        idx = FrenetFrame.find_nearest_point(1.1, 0.0, trajectory)
        assert idx == 1
        
        idx = FrenetFrame.find_nearest_point(0.1, 0.0, trajectory)
        assert idx == 0
    
    def test_wrap_angle(self):
        """Test angle wrapping."""
        assert np.isclose(FrenetFrame.wrap_angle(np.pi), np.pi)
        assert np.isclose(FrenetFrame.wrap_angle(-np.pi), -np.pi)
        assert np.isclose(FrenetFrame.wrap_angle(3*np.pi), np.pi)
        assert np.isclose(FrenetFrame.wrap_angle(-3*np.pi), -np.pi)
