import math

import numpy as np
import pytest

from itzi import snippets

num_cells_params = [10_000_000]
num_cells_ids = ["10M"]


def make_array_shape(num_cells: int) -> tuple[int, int]:
    side_length = int(math.sqrt(num_cells))
    return side_length, side_length


def make_positive_array(num_cells: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.random(size=make_array_shape(num_cells), dtype=np.float32) + np.float32(1e-6)


def make_signed_array(shape: tuple[int, int], scale: np.float32, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(size=shape, dtype=np.float32) * (np.float32(2.0) * scale) - scale


def annotate_math_benchmark(benchmark, *, num_cells: int, formula: str, math_path: str) -> None:
    benchmark.extra_info["lattice_updates"] = num_cells
    benchmark.extra_info["formula"] = formula
    benchmark.extra_info["math_path"] = math_path


def setup_almeida_args(num_cells: int) -> tuple[object, ...]:
    arr_shape = make_array_shape(num_cells)
    rng = np.random.default_rng(42)
    arr_hf = rng.random(size=arr_shape, dtype=np.float32) + np.float32(1e-4)
    arr_n = rng.random(size=arr_shape, dtype=np.float32) * np.float32(0.04) + np.float32(0.01)
    arr_qm1 = make_signed_array(arr_shape, np.float32(0.5), seed=43)
    arr_q0 = make_signed_array(arr_shape, np.float32(0.5), seed=44)
    arr_qp1 = make_signed_array(arr_shape, np.float32(0.5), seed=45)
    arr_q_norm = rng.random(size=arr_shape, dtype=np.float32) + np.float32(1e-4)
    arr_slope = make_signed_array(arr_shape, np.float32(0.005), seed=46)
    arr_out = np.empty_like(arr_hf)
    theta = np.float32(0.9)
    g = np.float32(9.81)
    dt = np.float32(0.25)
    return arr_hf, arr_n, arr_qm1, arr_q0, arr_qp1, arr_q_norm, arr_slope, arr_out, theta, g, dt


def setup_signed_flow_gms_args(num_cells: int) -> tuple[object, ...]:
    arr_shape = make_array_shape(num_cells)
    rng = np.random.default_rng(52)
    arr_flow_depth = rng.random(size=arr_shape, dtype=np.float32) + np.float32(1e-4)
    arr_n = rng.random(size=arr_shape, dtype=np.float32) * np.float32(0.04) + np.float32(0.01)
    arr_slope = make_signed_array(arr_shape, np.float32(0.02), seed=53)
    arr_out = np.empty_like(arr_flow_depth)
    max_slope = np.float32(0.01)
    return arr_flow_depth, arr_n, arr_slope, arr_out, max_slope


def setup_velocity_diagnostics_args(num_cells: int) -> tuple[object, ...]:
    arr_shape = make_array_shape(num_cells)
    rng = np.random.default_rng(62)
    arr_qx = make_signed_array(arr_shape, np.float32(0.2), seed=63)
    arr_qy = make_signed_array(arr_shape, np.float32(0.2), seed=64)
    arr_h = rng.random(size=arr_shape, dtype=np.float32) + np.float32(1e-4)
    arr_v = np.empty_like(arr_h)
    arr_vdir = np.empty_like(arr_h)
    arr_fr = np.empty_like(arr_h)
    g = np.float32(9.81)
    return arr_qx, arr_qy, arr_h, arr_v, arr_vdir, arr_fr, g


## velocity ##


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
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


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
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


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
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


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
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


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_pow_two_thirds(benchmark, num_cells):
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_pow_two_thirds,
        arr_h,
        arr_out,
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="pow(h, 2/3)",
        math_path="generic",
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_pow_two_thirds_float32(benchmark, num_cells: int) -> None:
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_pow_two_thirds_float32,
        arr_h,
        arr_out,
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="pow(h, 2/3)",
        math_path="float32",
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_cbrt_two_thirds(benchmark, num_cells):
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_cbrt_two_thirds,
        arr_h,
        arr_out,
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_pow_seven_thirds(benchmark, num_cells):
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_pow_seven_thirds,
        arr_h,
        arr_out,
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="pow(h, 7/3)",
        math_path="generic",
    )


@pytest.mark.parametrize(
    "num_cells",
    num_cells_params,
    ids=num_cells_ids,
)
def test_benchmark_pow_seven_thirds_float32(benchmark, num_cells: int) -> None:
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_pow_seven_thirds_float32,
        arr_h,
        arr_out,
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="pow(h, 7/3)",
        math_path="float32",
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_cbrt_seven_thirds(benchmark, num_cells):
    arr_h = make_positive_array(num_cells)
    arr_out = np.empty_like(arr_h)

    benchmark(
        snippets.arr_cbrt_seven_thirds,
        arr_h,
        arr_out,
    )


## generic vs explicit float32 math ##


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_almeida2013_generic(benchmark, num_cells: int) -> None:
    benchmark(
        snippets.arr_almeida2013_generic,
        *setup_almeida_args(num_cells),
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="solve_q almeida update",
        math_path="generic",
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_almeida2013_float32(benchmark, num_cells: int) -> None:
    benchmark(
        snippets.arr_almeida2013_float32,
        *setup_almeida_args(num_cells),
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="solve_q almeida update",
        math_path="float32",
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_signed_flow_gms_generic(benchmark, num_cells: int) -> None:
    benchmark(
        snippets.arr_signed_flow_gms_generic,
        *setup_signed_flow_gms_args(num_cells),
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="solve_q signed gms fallback",
        math_path="generic",
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_signed_flow_gms_float32(benchmark, num_cells: int) -> None:
    benchmark(
        snippets.arr_signed_flow_gms_float32,
        *setup_signed_flow_gms_args(num_cells),
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="solve_q signed gms fallback",
        math_path="float32",
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_velocity_diagnostics_generic(benchmark, num_cells: int) -> None:
    benchmark(
        snippets.arr_velocity_diagnostics_generic,
        *setup_velocity_diagnostics_args(num_cells),
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="solve_h velocity diagnostics",
        math_path="generic",
    )


@pytest.mark.parametrize("num_cells", num_cells_params, ids=num_cells_ids)
def test_benchmark_velocity_diagnostics_float32(benchmark, num_cells: int) -> None:
    benchmark(
        snippets.arr_velocity_diagnostics_float32,
        *setup_velocity_diagnostics_args(num_cells),
    )
    annotate_math_benchmark(
        benchmark,
        num_cells=num_cells,
        formula="solve_h velocity diagnostics",
        math_path="float32",
    )
