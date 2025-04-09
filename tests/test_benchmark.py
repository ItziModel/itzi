#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import math
import numpy as np
import pytest

from itzi.rasterdomain import RasterDomain
from itzi.const import DefaultValues
from itzi.surfaceflow import SurfaceFlowSimulation


def gen_eggbox(min_x, max_x, min_y, max_y, res, slope_x, slope_y,
               vshift, phase_shift, amplitude, period):
    """Return an eggbox 2D surface as a numpy array
    """
    X, Y = np.meshgrid(np.arange(min_x, max_x, res),
                       np.arange(min_y, max_y, res))
    ZX = vshift + slope_x*X + (amplitude/2) * np.sin(2*math.pi * (X-phase_shift) / period)
    ZY = slope_y*Y + (amplitude/2) * np.sin(2*math.pi * (Y-phase_shift) / period)
    return ZX + ZY


num_cells_params = [10_000, 1_000_000, 9_000_000]
cell_size_params = [1, 2, 5, 10]


@pytest.fixture(scope="class")
def eggbox_simulation(grass_xy_session, num_cells=10_000, cell_size=5):
    """Create the SurfaceFlow object
    """
    coord_min = 0
    coord_max = int(math.sqrt(num_cells)) * cell_size
    # Generate the Egg box DEM
    n_peaks = 5
    amplitude = 2
    slope_x = 0.001
    slope_y = 0.002
    period = coord_max / n_peaks
    phase_shift = period / 4
    egg_box = gen_eggbox(min_x=coord_min, max_x=coord_max,
                            min_y=coord_min, max_y=coord_max,
                            res=cell_size, slope_x=slope_x, slope_y=slope_y,
                            vshift=amplitude, phase_shift=phase_shift,
                            amplitude=amplitude, period=period)
    # Domain cover all the array
    array_shape = egg_box.shape
    mask = np.full(shape=array_shape, fill_value=False, dtype=np.bool_)
    manning = np.full(shape=array_shape, fill_value=0.03, dtype=np.float32)
    water_depth = np.full(shape=array_shape, fill_value=0.25, dtype=np.float32)
    sim_param = dict(dtmax=5, cfl=0.7, theta=0.9, hmin=0.005, vrouting=.1,
                        g=DefaultValues.G, slmax=DefaultValues.SLMAX)
    raster_domain = RasterDomain(dtype=np.float32, arr_mask=mask,
                                 cell_shape=(cell_size, cell_size))
    raster_domain.update_array('dem', egg_box)
    raster_domain.update_array('friction', manning)
    raster_domain.update_array('h', water_depth)
    surface_flow = SurfaceFlowSimulation(raster_domain, sim_param)
    surface_flow.update_flow_dir()
    # Spin up the model
    for i in range(10):
        surface_flow.solve_dt()
        surface_flow.step()
    return surface_flow


# @pytest.fixture(scope="class")
def benchmark_surface_flow(eggbox_simulation):
    for i in range(10):
        eggbox_simulation.solve_dt()
        eggbox_simulation.step()


def test_benchmark_surface_flow(benchmark, eggbox_simulation):
    benchmark(benchmark_surface_flow, eggbox_simulation)
