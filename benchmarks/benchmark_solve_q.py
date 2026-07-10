import math

import numpy as np
import pytest

from itzi.compute.partial_inertia_q import set_solve_q_tile_size, get_solve_q_tile_size, solve_q
from itzi.data_containers import SurfaceFlowParameters

NUM_CELLS_TO_SHAPE: dict[int, tuple[int, int]] = {
    1_000_000: (1_000, 1_000),
    10_000_000: (2_000, 5_000),
}

SOLVE_Q_TILE_SIZES: list[tuple[int, int]] = [
    (8, 256),
    (16, 256),
    (32, 256),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
    (256, 64),
]


def pad_array(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr, 1, mode="edge")


def zero_padded_array(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros((shape[0] + 2, shape[1] + 2), dtype=np.float32)


def setup_solve_q_args(num_cells: int) -> tuple:
    rows, cols = NUM_CELLS_TO_SHAPE[num_cells]
    shape = (rows, cols)
    params = SurfaceFlowParameters()
    dx = 5.0
    dy = 5.0
    starting_depth = np.float32(0.1)
    slope_x = np.float32(0.001)
    slope_y = np.float32(0.002)

    x = np.arange(cols, dtype=np.float32) * slope_x
    y = np.arange(rows, dtype=np.float32) * slope_y
    arr_z = pad_array(y[:, None] + x[None, :])
    arr_n = pad_array(np.full((rows, cols), 0.03, dtype=np.float32))
    arr_h = pad_array(np.full((rows, cols), starting_depth, dtype=np.float32))

    arr_qe = zero_padded_array(shape)
    arr_qs = zero_padded_array(shape)
    arr_hfe = zero_padded_array(shape)
    arr_hfs = zero_padded_array(shape)
    arr_bctype = zero_padded_array(shape)
    arr_qe_new = zero_padded_array(shape)
    arr_qs_new = zero_padded_array(shape)

    dt = min(params.dtmax, params.cfl * (min(dx, dy) / math.sqrt(params.g * starting_depth)))

    return (
        arr_z,
        arr_n,
        arr_h,
        arr_qe,
        arr_qs,
        arr_hfe,
        arr_hfs,
        arr_bctype,
        arr_qe_new,
        arr_qs_new,
        dt,
        dx,
        dy,
        params.g,
        params.theta,
        params.hmin,
        params.slope_threshold,
        params.max_slope,
    )


@pytest.mark.parametrize("num_cells", [1_000_000, 10_000_000], ids=["1M", "10M"])
@pytest.mark.parametrize(
    ("tile_rows", "tile_cols"),
    SOLVE_Q_TILE_SIZES,
    ids=[f"tile_{tile_rows}x{tile_cols}" for tile_rows, tile_cols in SOLVE_Q_TILE_SIZES],
)
def test_benchmark_solve_q(benchmark, num_cells: int, tile_rows: int, tile_cols: int) -> None:
    solve_q_args = setup_solve_q_args(num_cells)
    previous_tile_rows, previous_tile_cols = get_solve_q_tile_size()

    set_solve_q_tile_size(tile_rows, tile_cols)
    try:
        benchmark(solve_q, *solve_q_args)
    finally:
        set_solve_q_tile_size(previous_tile_rows, previous_tile_cols)

    benchmark.extra_info["lattice_updates"] = num_cells
    benchmark.extra_info["tile_rows"] = tile_rows
    benchmark.extra_info["tile_cols"] = tile_cols
