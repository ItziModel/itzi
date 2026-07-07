import math

import numpy as np
import pytest

from itzi import snippets

num_cells_params = [1_000_000, 100_000_000]


def make_positive_array(num_cells: int) -> np.ndarray:
    side_length = int(math.sqrt(num_cells))
    rng = np.random.default_rng()
    return rng.random(size=(side_length, side_length), dtype=np.float32) + np.float32(1e-6)


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
        snippets.branchless_velocity,
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
        snippets.branching_velocity,
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
        snippets.arr_hypot,
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
        snippets.arr_sqrt,
        arr_qe,
        arr_qs,
    )


## pow vs cbrt ##


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_pow_two_thirds(benchmark, num_cells):
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_pow_two_thirds,
        arr_h,
        arr_out,
    )


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_cbrt_two_thirds(benchmark, num_cells):
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_cbrt_two_thirds,
        arr_h,
        arr_out,
    )


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_pow_seven_thirds(benchmark, num_cells):
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_pow_seven_thirds,
        arr_h,
        arr_out,
    )


@pytest.mark.parametrize("num_cells", num_cells_params)
def test_benchmark_cbrt_seven_thirds(benchmark, num_cells):
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_cbrt_seven_thirds,
        arr_h,
        arr_out,
    )
