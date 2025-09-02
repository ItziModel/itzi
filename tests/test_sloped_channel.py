"""
Copyright (C) 2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt


from itzi.simulation_factories import create_memory_simulation
from itzi.rasterdomain import DomainData
from itzi.data_containers import SurfaceFlowParameters, SimulationConfig


def trapeze_channel(
    nx=200,
    ny=100,
    long_slope=0.01,  # slope in x-direction (m/m)
    lat_slope=0.1,  # slope of channel sides (z/|y|)
    flat_width=3,  # flat width at channel bottom (in cells)
    dx=10.0,  # cell resolution in x-direction (meters)
    dy=10.0,
):  # cell resolution in y-direction (meters)
    """
    Generate a synthetic 2D trapezoidal channel DEM.

    Parameters
    ----------
    nx, ny : int
        Grid size (x = longitudinal, y = lateral).
    long_slope : float
        Longitudinal slope (m/m).
    lat_slope : float
        Lateral slope of channel (m/m).
    flat_width : int
        Number of grid cells with flat bottom.
    dx, dy : float
        Cell resolution in x and y directions (meters).

    Returns
    -------
    Z : ndarray (ny, nx)
        Elevation surface in meters.
    """
    # Coordinates in meters
    x = np.arange(nx) * dx  # downstream distance in meters
    y = np.linspace(-ny // 2, ny // 2, ny) * dy  # lateral distance in meters

    X, Y = np.meshgrid(x, y)

    # Longitudinal slope to the east
    Z_long = 2000 - long_slope * X

    # V-shape with flat bottom: subtract half flat width in meters
    half_flat_meters = (flat_width / 2.0) * dy
    Z_lat = lat_slope * np.maximum(np.abs(Y) - half_flat_meters, 0)

    # Combine
    Z = Z_long + Z_lat

    return Z


def speed_GMS(flow_depth, n, slope):
    """Solve flow in m2/s with the Gauckler-Manning-Strickler formula."""
    # Hydraulics radius is flow_depth because the wetted perimeter is only the flow width, so it cancels out.
    v = (1.0 / n) * math.pow(flow_depth, 2.0 / 3.0) * math.sqrt(abs(slope))
    v = math.copysign(v, slope)
    return v


@pytest.mark.parametrize(
    "long_slope",
    [
        0.001,
        # 0.01,
        # 0.1,
        # 1,
        10,
        100,
    ],
)
@pytest.mark.parametrize(
    "lat_slope",
    [
        0.001,
        # 0.01,
        # 0.1,
        # 1,
        10,
        100,
    ],
)
@pytest.mark.parametrize(
    "dx, dy, cfl, dtmax",
    [
        (5, 5, 0.5, 2),
        (10, 10, 0.5, 3),
        (20, 20, 0.5, 5),
    ],
)
# @pytest.mark.parametrize(
#     "min_depth",
#     [
#         0.005, 0.01, 0.1
#     ]
# )
# @pytest.mark.parametrize(
#     "max_slope",
#     [
#         0.2, 0.5, 0.8, 1.0
#     ]
# )
# threshold=0.8 and max_slope=2 helps raising DtError.
def test_sloped_channel(
    test_data_temp_path,
    long_slope,
    lat_slope,
    dx,
    dy,
    cfl,
    dtmax,
    min_depth=0.005,
    slope_threshold=0.8,
    max_slope=0.8,
    rows=60,
    cols=63,
):
    surface_params = SurfaceFlowParameters(
        hmin=min_depth,
        cfl=cfl,
        # theta=,
        # vrouting=0.1,
        dtmax=dtmax,
        slope_threshold=slope_threshold,
        max_slope=max_slope,
        # max_error=self.sim_param["max_error"],
    )
    output_map_names = {
        "water_depth": "water_depth",
        "volume_error": "volume_error",
        "mean_boundary_flow": "mean_boundary_flow",
        "v": "v",
    }
    stats_file_name = (
        f"test_sloped_channel_stats_slope({long_slope},{lat_slope})_res({dx},{dy}).csv"
    )
    stats_file_path = Path(test_data_temp_path) / Path(stats_file_name)
    sim_config = SimulationConfig(
        start_time=datetime(year=2000, month=1, day=1, hour=0),
        end_time=datetime(year=2000, month=1, day=1, hour=1),
        record_step=timedelta(seconds=60 * 20),
        temporal_type="absolute",
        input_map_names={},
        output_map_names=output_map_names,
        stats_file=stats_file_path,
        surface_flow_parameters=surface_params,
        infiltration_model="null",
    )
    # Set arrays
    arr_dem = trapeze_channel(
        nx=cols, ny=rows, dx=dx, dy=dy, long_slope=long_slope, lat_slope=lat_slope
    )
    assert arr_dem.shape == (rows, cols)
    domain_data = DomainData(
        north=rows * dy, south=0, east=cols * dx, west=0, rows=rows, cols=cols
    )

    arr_rain_off = np.full_like(arr_dem, fill_value=0)
    arr_rain_on = np.full_like(arr_dem, fill_value=200 / (1000 * 3600))  # internal data in m/s
    arr_n = np.full_like(arr_dem, fill_value=0.05)
    # free eastmost boundary, 3 cells wide
    arr_bctype = np.ones_like(arr_dem)
    arr_bctype[:, -3:] = 2
    array_mask = np.full(shape=arr_dem.shape, fill_value=False, dtype=np.bool_)
    # Exclude 2 cells-wide border from the domain
    array_mask[:2, :] = True  # top 2 rows
    array_mask[-2:, :] = True  # bottom 2 rows
    array_mask[:, :2] = True  # left 2 columns
    array_mask[:, -2:] = True  # right 2 columns

    simulation = create_memory_simulation(
        sim_config=sim_config,
        domain_data=domain_data,
        arr_mask=array_mask,
        dtype=np.float32,
    )
    # Set the input arrays
    simulation.set_array("dem", arr_dem)  # Must be first
    simulation.set_array("bctype", arr_bctype)
    simulation.set_array("rain", arr_rain_off)
    simulation.set_array("friction", arr_n)

    # run the simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        # start rain 1 min after the start
        if simulation.sim_time == simulation.start_time + timedelta(seconds=1 * 60):
            simulation.set_array("rain", arr_rain_on)
        # Stop rain 10 min before the end
        if simulation.sim_time == simulation.end_time - timedelta(seconds=10 * 60):
            simulation.set_array("rain", arr_rain_off)
        simulation.update()
    simulation.finalize()
    # total_time_steps = simulation.time_steps_counters["since_start"]


def main():
    # Example
    Z = trapeze_channel(nx=60, ny=33, long_slope=0.1, lat_slope=10, flat_width=3)

    plt.figure(figsize=(8, 4))
    plt.imshow(Z, origin="lower", cmap="terrain", aspect="auto")
    plt.colorbar(label="Elevation (m)")
    plt.title("Synthetic V-shaped channel with flat bottom")
    plt.xlabel("Downstream (x)")
    plt.ylabel("Cross-stream (y)")
    plt.savefig("slope.png")

    flow_test = speed_GMS(flow_depth=1.00, n=0.1, slope=0.0)
    print(flow_test)


if __name__ == "__main__":
    main()
