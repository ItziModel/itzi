import math

import numpy as np
import pytest

from itzi import rastermetrics

num_cells_params = [1_000_000, 100_000_000]


## set_ext_array ##


def set_ext_array_np(
    arr_qext,
    arr_drain,
    arr_eff_precip,
    arr_ext,
):
    arr_ext[:] = arr_qext + arr_drain + arr_eff_precip


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_set_ext_array_cy(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    arr_qext = rng.random(size=arr_shape, dtype=np.float32)
    arr_drain = rng.random(size=arr_shape, dtype=np.float32)
    arr_eff_precip = rng.random(size=arr_shape, dtype=np.float32)
    arr_ext = rng.random(size=arr_shape, dtype=np.float32)

    benchmark(
        rastermetrics.set_ext_array,
        arr_qext,
        arr_drain,
        arr_eff_precip,
        arr_ext,
    )


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_set_ext_array_np(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    arr_qext = rng.random(size=arr_shape, dtype=np.float32)
    arr_drain = rng.random(size=arr_shape, dtype=np.float32)
    arr_eff_precip = rng.random(size=arr_shape, dtype=np.float32)
    arr_ext = rng.random(size=arr_shape, dtype=np.float32)

    benchmark(
        set_ext_array_np,
        arr_qext,
        arr_drain,
        arr_eff_precip,
        arr_ext,
    )


## calculate_total_volume ##


def calculate_total_volume_np(depth_array, cell_surface_area):
    return np.sum(depth_array) * cell_surface_area


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_calculate_total_volume_np(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    arr = rng.random(size=arr_shape, dtype=np.float32)
    cell_surface_area = 123.4

    benchmark(
        calculate_total_volume_np,
        arr,
        cell_surface_area,
    )


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_calculate_total_volume_cy(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    arr = rng.random(size=arr_shape, dtype=np.float32)
    cell_surface_area = 123.4

    benchmark(
        rastermetrics.calculate_total_volume,
        arr,
        cell_surface_area,
    )


## accumulate_rate_to_total ##


def accumulate_rate_to_total_np(accum_array, rate_array, time_delta_seconds):
    accum_array += rate_array * time_delta_seconds
    return None


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_accumulate_rate_to_total_np(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    accum_array = rng.random(size=arr_shape, dtype=np.float32)
    rate_array = rng.random(size=arr_shape, dtype=np.float32)
    time_delta_seconds = 1.23

    benchmark(
        accumulate_rate_to_total_np,
        accum_array,
        rate_array,
        time_delta_seconds,
    )


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_accumulate_rate_to_total_cy(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    accum_array = rng.random(size=arr_shape, dtype=np.float32)
    rate_array = rng.random(size=arr_shape, dtype=np.float32)
    time_delta_seconds = 1.23

    benchmark(
        rastermetrics.accumulate_rate_to_total,
        accum_array,
        rate_array,
        time_delta_seconds,
    )
