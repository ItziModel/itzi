from datetime import datetime, timedelta
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from itzi.simulation_factories import create_memory_simulation
from itzi.rasterdomain import DomainData
from itzi.data_containers import SurfaceFlowParameters, SimulationConfig
from itzi import messenger as msgr


# Define the parameter space - These are the parameters being optimized
parameter_space = [
    Real(0.2, 0.9, name="cfl"),  # CFL number
    Real(0.5, 10.0, name="dtmax"),  # Maximum time step length
    Real(0.001, 0.01, name="hmin"),  # Minimum depth (m)
    Real(0.5, 1.0, name="slope_threshold"),  # Slope threshold (m/m)
    Real(0.5, 10.0, name="max_slope"),  # Maximum slope when using Manning formula (m/m)
]

# Define parameter ranges for scenario generation
slope_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
res_values = [1, 5, 10, 30, 90]

Scenario = namedtuple(
    "Scenario",
    [
        "long_slope",
        "lat_slope",
        "dx",
        "dy",
    ],
)
OptimizedParameters = namedtuple(
    "OptimizedParameters", ["cfl", "dtmax", "hmin", "slope_threshold", "max_slope"]
)


def trapeze_channel(
    nx=200,
    ny=103,
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


def run_sloped_channel(
    long_slope,
    lat_slope,
    dx,
    dy,
    cfl,
    dtmax,
    min_depth=0.005,
    slope_threshold=0.8,
    max_slope=0.8,
    rows=100,
    cols=103,
):
    # Ensure exceptions are raised instead of calling sys.exit()
    msgr.raise_on_error = True
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

    sim_config = SimulationConfig(
        start_time=datetime(year=2000, month=1, day=1, hour=0),
        end_time=datetime(year=2000, month=1, day=1, hour=2),
        record_step=timedelta(seconds=60 * 20),
        temporal_type="absolute",
        input_map_names={},
        output_map_names=output_map_names,
        stats_file=None,
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
    rain_set = False
    while simulation.sim_time < simulation.end_time:
        # start rain 30 seconds after the start
        if not rain_set and simulation.sim_time >= simulation.start_time + timedelta(seconds=30):
            simulation.set_array("rain", arr_rain_on)
            rain_set = True
        simulation.update()
    simulation.finalize()
    return simulation.time_steps_counters["since_start"]


def optimize_for_scenario(scenario: Scenario):
    """
    Optimize parameters for a specific scenario.

    Inputs (fixed): long_slope, lat_slope, dx, dy
    Outputs (optimized): cfl, dtmax, hmin, slope_threshold, max_slope
    """

    @use_named_args(parameter_space)
    def objective(cfl, dtmax, hmin, slope_threshold, max_slope):
        """
        Objective function that runs simulation with:
        - Fixed scenario: long_slope, lat_slope, dx, dy
        - Variable parameters: cfl, dtmax, hmin, slope_threshold, max_slope
        """
        try:
            # Run simulation with FIXED scenario and OPTIMIZED parameters
            time_steps = run_sloped_channel(
                long_slope=scenario.long_slope,
                lat_slope=scenario.lat_slope,
                dx=scenario.dx,
                dy=scenario.dy,
                cfl=cfl,
                dtmax=dtmax,
                min_depth=hmin,
                slope_threshold=slope_threshold,
                max_slope=max_slope,
            )
            return time_steps  # Minimize number of time steps

        except Exception:
            return 1e12  # Large penalty for unstable simulations

    # Run optimization for this specific scenario
    result = gp_minimize(
        func=objective,
        dimensions=parameter_space,
        n_calls=50,
        n_initial_points=10,
        random_state=42,
    )

    # Return optimized parameters as dictionary
    return OptimizedParameters(
        cfl=result.x[0],
        dtmax=result.x[1],
        hmin=result.x[2],
        slope_threshold=result.x[3],
        max_slope=result.x[4],
    )


def create_scenario_matrix_full():
    """
    Create full Cartesian product of all parameter combinations.
    Warning: This creates len(long_slope) * len(lat_slope) * len(dx) * len(dy) scenarios!
    """
    scenarios = []
    for long_slope, lat_slope, dx, dy in itertools.product(
        slope_values, slope_values, res_values, res_values
    ):
        scenario = Scenario(long_slope, lat_slope, dx, dy)
        scenarios.append(scenario)
    return scenarios


def create_scenario_matrix_full_equals():
    """
    Create full Cartesian product of all parameter combinations.
    Warning: This creates len(long_slope) * len(lat_slope) * len(dx) * len(dy) scenarios!
    """
    scenarios = []
    for (
        long_slope,
        dx,
    ) in itertools.product(slope_values, res_values):
        scenarios.append(Scenario(long_slope, long_slope, dx, dx))
    return scenarios


def save_results_parquet(optimal_params_lookup, filename="optimization_results.parquet"):
    """Save optimization results with each scenario parameter as separate column"""
    results = []
    for scenario, params in optimal_params_lookup.items():
        result = {
            # Scenario parameters (inputs)
            "long_slope": scenario.long_slope,
            "lat_slope": scenario.lat_slope,
            "dx": scenario.dx,
            "dy": scenario.dy,
            # Optimized parameters (outputs)
            "cfl": params.cfl,
            "dtmax": params.dtmax,
            "hmin": params.hmin,
            "slope_threshold": params.slope_threshold,
            "max_slope": params.max_slope,
        }
        results.append(result)

    df = pd.DataFrame(results)
    df.to_parquet(filename, index=False)
    return filename


def main():
    # scenarios = create_scenario_matrix_full()  # Warning: Very large!
    scenarios = create_scenario_matrix_full_equals()
    # scenarios = [Scenario(long_slope=0.001, lat_slope=0.001, dx=30, dy=30)]

    print(f"Generated {len(scenarios)} scenarios")
    # print("First few scenarios:")
    # for i, scenario in enumerate(scenarios[:50]):
    #     print(f"  {i + 1}: {scenario}")

    optimal_params_lookup = {}
    for scenario in scenarios:
        print(f"Optimizing for scenario: {scenario}")
        optimal_params = optimize_for_scenario(scenario)

        # Store in lookup table. Key is scenario
        optimal_params_lookup[scenario] = optimal_params
        print(f"  Optimal parameters: {optimal_params}")
    # Save to parquet
    filename = save_results_parquet(optimal_params_lookup)
    # load from parquet
    df_results = pd.read_parquet(filename)
    print(df_results)


if __name__ == "__main__":
    main()
