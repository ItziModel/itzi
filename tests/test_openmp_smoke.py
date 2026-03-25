from __future__ import annotations

import numpy as np

from itzi import flow, rastermetrics


OPENMP_SMOKE_SHAPE = (1024, 1024)


def _full_array(value: float) -> np.ndarray:
    arr = np.full(OPENMP_SMOKE_SHAPE, value, dtype=np.float32)
    assert arr.flags.c_contiguous
    return arr


def test_flow_apply_hydrology_large_contiguous_arrays():
    arr_rain = _full_array(0.002)
    arr_inf = _full_array(0.0005)
    arr_capped_losses = _full_array(0.00025)
    arr_h = _full_array(0.01)
    arr_eff_precip = np.empty(OPENMP_SMOKE_SHAPE, dtype=np.float32)
    dt = 60.0

    flow.apply_hydrology(
        arr_rain=arr_rain,
        arr_inf=arr_inf,
        arr_capped_losses=arr_capped_losses,
        arr_h=arr_h,
        arr_eff_precip=arr_eff_precip,
        dt=dt,
    )

    expected = np.maximum(-arr_h / dt, arr_rain - arr_inf - arr_capped_losses)
    np.testing.assert_allclose(arr_eff_precip, expected)


def test_flow_branchless_velocity_large_contiguous_arrays():
    arr_qe = _full_array(0.5)
    arr_qs = _full_array(0.25)
    arr_hfe = _full_array(0.1)
    arr_hfs = _full_array(0.2)

    flow.branchless_velocity(
        arr_qe=arr_qe,
        arr_qs=arr_qs,
        arr_hfe=arr_hfe,
        arr_hfs=arr_hfs,
    )


def test_rastermetrics_calculate_wse_large_contiguous_arrays():
    h_array = _full_array(0.3)
    dem_array = np.arange(np.prod(OPENMP_SMOKE_SHAPE), dtype=np.float32).reshape(
        OPENMP_SMOKE_SHAPE
    )
    assert dem_array.flags.c_contiguous

    result = rastermetrics.calculate_wse(h_array, dem_array)

    np.testing.assert_allclose(result, h_array + dem_array)


def test_rastermetrics_calculate_flux_large_contiguous_arrays():
    flow_array = np.arange(np.prod(OPENMP_SMOKE_SHAPE), dtype=np.float32).reshape(
        OPENMP_SMOKE_SHAPE
    )
    assert flow_array.flags.c_contiguous
    cell_size = 2.5

    result = rastermetrics.calculate_flux(flow_array, cell_size)

    np.testing.assert_allclose(result, flow_array * cell_size)


def test_rastermetrics_accumulate_rate_to_total_large_contiguous_arrays():
    accum_array = _full_array(1.0)
    rate_array = _full_array(0.25)
    time_delta_seconds = 12.0

    rastermetrics.accumulate_rate_to_total(
        accum_array=accum_array,
        rate_array=rate_array,
        time_delta_seconds=time_delta_seconds,
        padded=False,
    )

    np.testing.assert_allclose(accum_array, _full_array(1.0 + 0.25 * time_delta_seconds))
