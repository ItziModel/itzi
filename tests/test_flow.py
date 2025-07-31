#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for flow.pyx optimizations
Testing mathematical calculations: velocity magnitude and direction
"""

import numpy as np
import pytest
from math import atan2, pi

import itzi.flow as flow


def test_velocity_direction_calculation():
    """Test velocity direction calculation"""
    # Test cases: (vx, vy, expected_direction_degrees)
    test_cases = [
        (1.0, 0.0, 0.0),  # East
        (0.0, 1.0, 270.0),  # North (note: -vy in atan2)
        (-1.0, 0.0, 180.0),  # West
        (0.0, -1.0, 90.0),  # South
        (1.0, 1.0, 315.0),  # Northeast
        (-1.0, 1.0, 225.0),  # Northwest
        (1.0, -1.0, 45.0),  # Southeast
        (-1.0, -1.0, 135.0),  # Southwest
    ]
    for vx, vy, expected_deg in test_cases:
        # Calculate direction as in solve_h
        vdir = atan2(-vy, vx) * 180.0 / pi
        vdir = vdir + 360.0 * (vdir < 0)
        assert vdir == pytest.approx(expected_deg)


def test_vectorizable_velocity_calculation():
    """Test that vectorizable velocity calculation gives correct results"""
    eps = 1e-12
    # Test cases: (flow, flow_depth, expected_velocity)
    test_cases = [
        (2.0, 1.0, 2.0),  # Normal case
        (0.0, 1.0, 0.0),  # No flow
        (2.0, 0.0, 0.0),  # No depth (should be zero after masking)
        (2.0, -0.1, 0.0),  # Negative depth (should be zero after masking)
        (5.0, 2.5, 2.0),  # Another normal case
    ]
    for q, hf, expected_v in test_cases:
        # Original conditional approach
        if hf <= 0.0:
            v_original = 0.0
        else:
            v_original = q / hf

        # Optimized branchless approach
        v_optimized = q / max(hf, eps) * (hf > 0)
        assert v_original == pytest.approx(v_optimized)
        assert v_optimized == pytest.approx(expected_v)


class TestWaterDepthFunction:
    """Integration tests for flow functions with optimized calculations"""

    def setup_method(self):
        """Set up test arrays"""
        self.shape = (5, 5)
        self.dtype = np.float64

        # Create test arrays
        self.arr_ext = np.zeros(self.shape, dtype=self.dtype)
        self.arr_qe = np.ones(self.shape, dtype=self.dtype) * 0.5
        self.arr_qs = np.ones(self.shape, dtype=self.dtype) * 0.3
        self.arr_bct = np.zeros(self.shape, dtype=self.dtype)
        self.arr_bcv = np.zeros(self.shape, dtype=self.dtype)
        self.arr_h = np.ones(self.shape, dtype=self.dtype) * 0.1
        self.arr_hmax = np.ones(self.shape, dtype=self.dtype) * 0.1
        self.arr_hfix = np.zeros(self.shape, dtype=self.dtype)
        self.arr_herr = np.zeros(self.shape, dtype=self.dtype)
        self.arr_hfe = np.ones(self.shape, dtype=self.dtype) * 0.05
        self.arr_hfs = np.ones(self.shape, dtype=self.dtype) * 0.05
        self.arr_v = np.zeros(self.shape, dtype=self.dtype)
        self.arr_vdir = np.zeros(self.shape, dtype=self.dtype)
        self.arr_vmax = np.zeros(self.shape, dtype=self.dtype)
        self.arr_fr = np.zeros(self.shape, dtype=self.dtype)

        # Parameters
        self.dx = 1.0
        self.dy = 1.0
        self.dt = 0.1
        self.g = 9.81

    def test_solve_h_velocity_calculations(self):
        """Test that solve_h produces reasonable velocity calculations"""
        # Run solve_h
        flow.solve_h(
            arr_ext=self.arr_ext,
            arr_qe=self.arr_qe,
            arr_qs=self.arr_qs,
            arr_bct=self.arr_bct,
            arr_bcv=self.arr_bcv,
            arr_h=self.arr_h,
            arr_hmax=self.arr_hmax,
            arr_hfix=self.arr_hfix,
            arr_herr=self.arr_herr,
            arr_hfe=self.arr_hfe,
            arr_hfs=self.arr_hfs,
            arr_v=self.arr_v,
            arr_vdir=self.arr_vdir,
            arr_vmax=self.arr_vmax,
            arr_fr=self.arr_fr,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
        )

        # Check that velocities are reasonable
        # Interior cells should have non-zero velocities
        interior_v = self.arr_v[1:-1, 1:-1]
        assert np.all(interior_v >= 0), "Velocities should be non-negative"
        assert np.any(interior_v > 0), "Some interior velocities should be positive"

        # Check that Froude numbers are reasonable
        interior_fr = self.arr_fr[1:-1, 1:-1]
        assert np.all(interior_fr >= 0), "Froude numbers should be non-negative"
        assert np.all(interior_fr < 15), "Froude numbers should be reasonable"

        # Check that velocity directions are in valid range [0, 360)
        interior_vdir = self.arr_vdir[1:-1, 1:-1]
        assert np.all(interior_vdir >= 0), "Velocity directions should be >= 0"
        assert np.all(interior_vdir < 360), "Velocity directions should be < 360"

    def test_solve_h_with_zero_flow_depths(self):
        """Test solve_h behavior with zero flow depths"""
        # Set some flow depths to zero
        self.arr_hfe[2, 2] = 0.0
        self.arr_hfs[2, 2] = 0.0

        # Run solve_h
        flow.solve_h(
            arr_ext=self.arr_ext,
            arr_qe=self.arr_qe,
            arr_qs=self.arr_qs,
            arr_bct=self.arr_bct,
            arr_bcv=self.arr_bcv,
            arr_h=self.arr_h,
            arr_hmax=self.arr_hmax,
            arr_hfix=self.arr_hfix,
            arr_herr=self.arr_herr,
            arr_hfe=self.arr_hfe,
            arr_hfs=self.arr_hfs,
            arr_v=self.arr_v,
            arr_vdir=self.arr_vdir,
            arr_vmax=self.arr_vmax,
            arr_fr=self.arr_fr,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
        )

        # Should not crash and should produce finite results
        assert np.all(np.isfinite(self.arr_v)), "All velocities should be finite"
        assert np.all(np.isfinite(self.arr_fr)), "All Froude numbers should be finite"
        assert np.all(np.isfinite(self.arr_vdir)), "All velocity directions should be finite"


class TestFixedWaterLevel:
    """Test if the boundary condition type 4 (fixed water level) is properly applied"""

    def setup_method(self):
        """Set up test arrays"""
        self.shape = (5, 5)
        self.dtype = np.float64

        # Create test arrays
        self.arr_ext = np.zeros(self.shape, dtype=self.dtype)
        self.arr_qe = np.zeros(self.shape, dtype=self.dtype)
        self.arr_qs = np.zeros(self.shape, dtype=self.dtype)
        self.arr_herr = np.zeros(self.shape, dtype=self.dtype)
        self.arr_hfe = np.ones(self.shape, dtype=self.dtype) * 0.05
        self.arr_hfs = np.ones(self.shape, dtype=self.dtype) * 0.05
        self.arr_v = np.zeros(self.shape, dtype=self.dtype)
        self.arr_vdir = np.zeros(self.shape, dtype=self.dtype)
        self.arr_vmax = np.zeros(self.shape, dtype=self.dtype)
        self.arr_fr = np.zeros(self.shape, dtype=self.dtype)

        # Parameters
        self.dx = 1.0
        self.dy = 1.0
        self.dt = 0.1
        self.g = 9.81

        # fixed boundary on center cell
        bct_values = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.arr_bct = np.array(bct_values, dtype=self.dtype)
        assert self.shape == self.arr_bct.shape

    def test_adding_water(self):
        bcv_values = [
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.0, 0, 0],
            [0, 0, 1.5, 0, 0],
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.0, 0, 0],
        ]
        arr_bcv = np.array(bcv_values, dtype=self.dtype)
        arr_h = np.ones(self.shape, dtype=self.dtype)
        arr_hmax = np.ones(self.shape, dtype=self.dtype)
        arr_hfix = np.zeros(self.shape, dtype=self.dtype)
        assert arr_bcv.shape == self.shape

        flow.solve_h(
            arr_ext=self.arr_ext,
            arr_qe=self.arr_qe,
            arr_qs=self.arr_qs,
            arr_bct=self.arr_bct,
            arr_bcv=arr_bcv,
            arr_h=arr_h,
            arr_hmax=arr_hmax,
            arr_hfix=arr_hfix,
            arr_herr=self.arr_herr,
            arr_hfe=self.arr_hfe,
            arr_hfs=self.arr_hfs,
            arr_v=self.arr_v,
            arr_vdir=self.arr_vdir,
            arr_vmax=self.arr_vmax,
            arr_fr=self.arr_fr,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
        )
        assert np.max(arr_hmax) == pytest.approx(1.5)
        assert np.sum(arr_hfix) == pytest.approx(0.5)
        assert np.max(arr_h) == pytest.approx(1.5)
        assert np.min(arr_h) == pytest.approx(1.0)

    def test_removing_water(self):
        bcv_values = [
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.5, 0, 0],
            [0, 0, 0.0, 0, 0],
            [0, 0, 0.0, 0, 0],
        ]
        arr_bcv = np.array(bcv_values, dtype=self.dtype)
        arr_h = np.ones(self.shape, dtype=self.dtype)
        arr_hmax = np.ones(self.shape, dtype=self.dtype)
        arr_hfix = np.zeros(self.shape, dtype=self.dtype)
        assert arr_bcv.shape == self.shape

        flow.solve_h(
            arr_ext=self.arr_ext,
            arr_qe=self.arr_qe,
            arr_qs=self.arr_qs,
            arr_bct=self.arr_bct,
            arr_bcv=arr_bcv,
            arr_h=arr_h,
            arr_hmax=arr_hmax,
            arr_hfix=arr_hfix,
            arr_herr=self.arr_herr,
            arr_hfe=self.arr_hfe,
            arr_hfs=self.arr_hfs,
            arr_v=self.arr_v,
            arr_vdir=self.arr_vdir,
            arr_vmax=self.arr_vmax,
            arr_fr=self.arr_fr,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
        )
        assert np.max(arr_hmax) == pytest.approx(1.0)
        assert np.sum(arr_hfix) == pytest.approx(-0.5)
        assert np.max(arr_h) == pytest.approx(1.0)
        assert np.min(arr_h) == pytest.approx(0.5)
