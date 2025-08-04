import math

import numpy as np
import pytest

from itzi import flow

num_cells_params = [1_000_000, 100_000_000]


## velocity ##


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_velocity_branchless(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    arr_qe = rng.random(size=arr_shape, dtype=np.float32)
    arr_qs = rng.random(size=arr_shape, dtype=np.float32)
    arr_hfe = rng.random(size=arr_shape, dtype=np.float32)
    arr_hfs = rng.random(size=arr_shape, dtype=np.float32)

    benchmark(
        flow.branchless_velocity,
        arr_qe,
        arr_qs,
        arr_hfe,
        arr_hfs,
    )


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_velocity_branching(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    arr_qe = rng.random(size=arr_shape, dtype=np.float32)
    arr_qs = rng.random(size=arr_shape, dtype=np.float32)
    arr_hfe = rng.random(size=arr_shape, dtype=np.float32)
    arr_hfs = rng.random(size=arr_shape, dtype=np.float32)

    benchmark(
        flow.branching_velocity,
        arr_qe,
        arr_qs,
        arr_hfe,
        arr_hfs,
    )


## hypot vs sqrt ##


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_hypot(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    arr_qe = rng.random(size=arr_shape, dtype=np.float32)
    arr_qs = rng.random(size=arr_shape, dtype=np.float32)

    benchmark(
        flow.arr_hypot,
        arr_qe,
        arr_qs,
    )


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_sqrt(benchmark, num_cells):
    side_length = int(math.sqrt(num_cells))
    arr_shape = (side_length, side_length)
    rng = np.random.default_rng()
    arr_qe = rng.random(size=arr_shape, dtype=np.float32)
    arr_qs = rng.random(size=arr_shape, dtype=np.float32)

    benchmark(
        flow.arr_sqrt,
        arr_qe,
        arr_qs,
    )
