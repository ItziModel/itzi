import math

import numpy as np
import pytest

from itzi.compute.partial_inertia_h import set_solve_h_tile_size, get_solve_h_tile_size, solve_h
from itzi.data_containers import SurfaceFlowParameters

NUM_CELLS_TO_SHAPE: dict[int, tuple[int, int]] = {
    1_000_000: (1_000, 1_000),
    10_000_000: (2_000, 5_000),
}

UPDATE_H_TILE_SIZES: list[tuple[int, int]] = [
    (8, 256),
    (16, 256),
    (32, 256),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
    (256, 64),
]


def full_padded_array(shape: tuple[int, int], fill_value: np.float32) -> np.ndarray:
    arr = np.empty((shape[0] + 2, shape[1] + 2), dtype=np.float32)
    arr.fill(fill_value)
    return arr


def zero_padded_array(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros((shape[0] + 2, shape[1] + 2), dtype=np.float32)


def setup_update_h_args(num_cells: int) -> tuple:
    rows, cols = NUM_CELLS_TO_SHAPE[num_cells]
    shape = (rows, cols)
    params = SurfaceFlowParameters()
    dx = 5.0
    dy = 5.0
    starting_depth = np.float32(0.1)

    arr_h = full_padded_array(shape, starting_depth)
    arr_hmax = arr_h.copy()

    arr_ext = zero_padded_array(shape)
    arr_qe = full_padded_array(shape, np.float32(0.01))
    arr_qs = full_padded_array(shape, np.float32(0.01))
    arr_bct = zero_padded_array(shape)
    arr_bcv = zero_padded_array(shape)
    arr_hfe = full_padded_array(shape, starting_depth)
    arr_hfs = full_padded_array(shape, starting_depth)

    arr_hfix = zero_padded_array(shape)
    arr_herr = zero_padded_array(shape)
    arr_v = zero_padded_array(shape)
    arr_vdir = zero_padded_array(shape)
    arr_vmax = zero_padded_array(shape)
    arr_fr = zero_padded_array(shape)

    dt = min(params.dtmax, params.cfl * (min(dx, dy) / math.sqrt(params.g * starting_depth)))

    return (
        arr_ext,
        arr_qe,
        arr_qs,
        arr_bct,
        arr_bcv,
        arr_h,
        arr_hmax,
        arr_hfix,
        arr_herr,
        arr_hfe,
        arr_hfs,
        arr_v,
        arr_vdir,
        arr_vmax,
        arr_fr,
        dx,
        dy,
        dt,
        params.g,
    )


@pytest.mark.parametrize("num_cells", [1_000_000, 10_000_000], ids=["1M", "10M"])
@pytest.mark.parametrize(
    ("tile_rows", "tile_cols"),
    UPDATE_H_TILE_SIZES,
    ids=[f"tile_{tile_rows}x{tile_cols}" for tile_rows, tile_cols in UPDATE_H_TILE_SIZES],
)
def test_benchmark_update_h(benchmark, num_cells: int, tile_rows: int, tile_cols: int) -> None:
    solve_h_args = setup_update_h_args(num_cells)
    previous_tile_rows, previous_tile_cols = get_solve_h_tile_size()

    set_solve_h_tile_size(tile_rows, tile_cols)
    try:
        benchmark(solve_h, *solve_h_args)
    finally:
        set_solve_h_tile_size(previous_tile_rows, previous_tile_cols)

    benchmark.extra_info["lattice_updates"] = num_cells
    benchmark.extra_info["tile_rows"] = tile_rows
    benchmark.extra_info["tile_cols"] = tile_cols
