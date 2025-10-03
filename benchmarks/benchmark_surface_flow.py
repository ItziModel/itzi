#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" """

import math
import datetime

import numpy as np
import pytest

from itzi.data_containers import SurfaceFlowParameters
from itzi.rasterdomain import RasterDomain
from itzi.surfaceflow import SurfaceFlowSimulation


def gen_eggbox(
    min_x,
    max_x,
    min_y,
    max_y,
    res,
    slope_x,
    slope_y,
    vshift,
    phase_shift,
    amplitude,
    period,
):
    """Return an eggbox 2D surface as a numpy array"""
    X, Y = np.meshgrid(np.arange(min_x, max_x, res), np.arange(min_y, max_y, res))
    ZX = vshift + slope_x * X + (amplitude / 2) * np.sin(2 * math.pi * (X - phase_shift) / period)
    ZY = slope_y * Y + (amplitude / 2) * np.sin(2 * math.pi * (Y - phase_shift) / period)
    return ZX + ZY


num_cells_params = [10_000, 100_000, 1_000_000, 10_000_000]
cell_size_params = [1, 2, 5, 10, 20]


def setup_eggbox_simulation(num_cells=10_000, cell_size=5):
    """Create the SurfaceFlow object"""
    starting_depth = 0.1
    coord_min = 0
    coord_max = int(math.sqrt(num_cells)) * cell_size
    # Generate the Egg box DEM
    n_peaks = 5
    amplitude = 2
    slope_x = 0.001
    slope_y = 0.002
    period = coord_max / n_peaks
    phase_shift = period / 4
    egg_box = gen_eggbox(
        min_x=coord_min,
        max_x=coord_max,
        min_y=coord_min,
        max_y=coord_max,
        res=cell_size,
        slope_x=slope_x,
        slope_y=slope_y,
        vshift=amplitude,
        phase_shift=phase_shift,
        amplitude=amplitude,
        period=period,
    )
    # Domain cover all the array
    array_shape = egg_box.shape
    mask = np.full(shape=array_shape, fill_value=False, dtype=np.bool_)
    manning = np.full(shape=array_shape, fill_value=0.03, dtype=np.float32)
    water_depth = np.full(shape=array_shape, fill_value=starting_depth, dtype=np.float32)
    sim_param = SurfaceFlowParameters()  # use default
    raster_domain = RasterDomain(
        dtype=np.float32, arr_mask=mask, cell_shape=(cell_size, cell_size)
    )
    raster_domain.update_array("dem", egg_box)
    raster_domain.update_array("friction", manning)
    raster_domain.update_array("water_depth", water_depth)
    surface_flow = SurfaceFlowSimulation(raster_domain, sim_param)
    surface_flow.update_flow_dir()
    # Spin up the model
    for i in range(5):
        surface_flow.solve_dt()
        surface_flow.step()
    return surface_flow


def benchmark_surface_flow_n_steps(eggbox_simulation, n_steps=10):
    for _ in range(n_steps):
        eggbox_simulation.solve_dt()
        eggbox_simulation.step()
    return n_steps


def benchmark_surface_flow_n_seconds(eggbox_simulation, n_seconds=30):
    time_left = datetime.timedelta(seconds=n_seconds)
    n_steps = 0
    while time_left >= datetime.timedelta(seconds=0):
        eggbox_simulation.solve_dt()
        eggbox_simulation.step()
        time_left -= eggbox_simulation.dt
        n_steps += 1
    return n_steps


@pytest.mark.parametrize("num_cells", num_cells_params)
@pytest.mark.parametrize("cell_size", [5])  # Set as parameter to get it in the output json
@pytest.mark.parametrize("n_steps", [5, 10])
def test_benchmark_surface_flow_n_steps(benchmark, num_cells, cell_size, n_steps):
    """Run the benchmark for a given number of cells and cell size"""
    eggbox_sim = setup_eggbox_simulation(num_cells=num_cells, cell_size=cell_size)
    benchmark(benchmark_surface_flow_n_steps, eggbox_sim, n_steps)
    benchmark.extra_info["lattice_updates"] = n_steps * num_cells


@pytest.mark.parametrize("num_cells", num_cells_params)
@pytest.mark.parametrize("cell_size", cell_size_params)
@pytest.mark.parametrize("n_seconds", [30])  # Set as parameter to get it in the output json
def test_benchmark_surface_flow_n_seconds(benchmark, num_cells, cell_size, n_seconds):
    """Run the benchmark for a given number of cells and cell size"""
    results = []

    def wrapper(eggbox_sim, n_seconds):
        n_steps = benchmark_surface_flow_n_seconds(eggbox_sim, n_seconds)
        results.append(n_steps)
        return n_steps

    eggbox_sim = setup_eggbox_simulation(num_cells=num_cells, cell_size=cell_size)
    n_steps = benchmark(wrapper, eggbox_sim, n_seconds)
    # Calculate statistics
    n_steps_mean = np.mean(results)
    n_steps_std = np.std(results)
    print(f"Number of steps: {n_steps} (mean: {n_steps_mean:.1f} ± {n_steps_std:.1f})")
    benchmark.extra_info["n_steps_mean"] = n_steps_mean
    benchmark.extra_info["n_steps_std"] = n_steps_std
    benchmark.extra_info["lattice_updates"] = n_steps_mean * num_cells
