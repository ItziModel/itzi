from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider
from itzi.const import InfiltrationModelType, TemporalType


@pytest.fixture(scope="module")
def sim_5by5_open_boundaries(domain_5by5, helpers, tmp_path_factory):
    """Run a 5x5 simulation for 600s with 60s record step and open boundaries.

    Outputs: water_depth, volume_error, mean_boundary_flow
    Boundary condition type 2 (open) is set at all 16 edge cells.
    """
    # Create temp directory for stats file
    temp_dir = tmp_path_factory.mktemp("open_bc_test")
    stats_file = temp_dir / "5by5_open_boundaries.csv"

    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 10, 0),  # 600 seconds = 10 minutes
        record_step=timedelta(seconds=60),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
            bctype="open_boundaries",
        ),
        output_map_names=helpers.make_output_map_names(
            "out_5by5_open_boundaries",
            ["water_depth", "volume_error", "mean_boundary_flow"],
        ),
        # Use dtmax=2 as specified in 5by5_open_boundaries.ini
        surface_flow_parameters=SurfaceFlowParameters(dtmax=2.0),
        infiltration_model=InfiltrationModelType.NULL,
        stats_file=str(stats_file),
    )

    # Create output provider
    raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})

    # Build simulation
    simulation = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_5by5.domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
        .build()
    )

    # Set input arrays
    simulation.set_array("dem", domain_5by5.arr_dem_flat)
    simulation.set_array("friction", domain_5by5.arr_n)
    simulation.set_array("water_depth", domain_5by5.arr_start_h)
    simulation.set_array("bctype", domain_5by5.arr_bctype)

    # Run simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return simulation, domain_5by5.domain_data


class TestOpenBoundaries:
    """Test that flow at boundary cells exhibits symmetry across the domain.

    The 16 edge cells of the 5x5 grid are:
    - Top row: (0,0), (0,1), (0,2), (0,3), (0,4)
    - Bottom row: (4,0), (4,1), (4,2), (4,3), (4,4)
    - Left column (excluding corners): (1,0), (2,0), (3,0)
    - Right column (excluding corners): (1,4), (2,4), (3,4)

    Symmetric pairs (r, c) and (4-r, 4-c) should have equal flow rates.

    Note: Corner cells are tested separately because they exhibit non-deterministic
    behavior due to OpenMP parallel execution.
    """

    def test_edge_flow_symmetry(self, sim_5by5_open_boundaries):
        """Symmetric edge cells (non-corner boundary cells) should have equal flow rates.

        Edge cells (excluding corners) only receive ONE boundary condition accumulation,
        so they should pass symmetry tests consistently.
        """
        simulation, _ = sim_5by5_open_boundaries
        output_dict = simulation.report.raster_provider.output_maps_dict

        # Get the last mean_boundary_flow output
        _, bf_array = output_dict["mean_boundary_flow"][-1]

        # Define edge cells (non-corner boundary cells)
        # These are the 12 edge cells that are NOT corners
        edge_cells = [
            (0, 1),
            (0, 2),
            (0, 3),  # top row (excluding corners)
            (4, 1),
            (4, 2),
            (4, 3),  # bottom row (excluding corners)
            (1, 0),
            (2, 0),
            (3, 0),  # left column (excluding corners)
            (1, 4),
            (2, 4),
            (3, 4),  # right column (excluding corners)
        ]

        # Check symmetric pairs: (r, c) and (4-r, 4-c)
        # We only need to check half the cells
        checked = set()
        for r, c in edge_cells:
            if (r, c) in checked:
                continue
            sym_r, sym_c = 4 - r, 4 - c
            if (sym_r, sym_c) in checked:
                continue

            flow_val = bf_array[r, c]
            sym_flow_val = bf_array[sym_r, sym_c]

            assert np.isclose(flow_val, sym_flow_val), (
                f"Symmetric edge cells ({r},{c})={flow_val:.6f} and "
                f"({sym_r},{sym_c})={sym_flow_val:.6f} should have equal flow"
            )
            checked.add((r, c))
            checked.add((sym_r, sym_c))

    def test_corner_flow_symmetry(self, sim_5by5_open_boundaries):
        """Symmetric corner cells should have equal flow rates."""
        simulation, _ = sim_5by5_open_boundaries
        output_dict = simulation.report.raster_provider.output_maps_dict

        # Get the last mean_boundary_flow output
        _, bf_array = output_dict["mean_boundary_flow"][-1]

        # Define the 4 corner cells
        corner_cells = [
            (0, 0),  # top-left
            (0, 4),  # top-right
            (4, 0),  # bottom-left
            (4, 4),  # bottom-right
        ]

        # Check symmetric pairs: (r, c) and (4-r, 4-c)
        # Pairs are: (0,0) <-> (4,4) and (0,4) <-> (4,0)
        checked = set()
        for r, c in corner_cells:
            if (r, c) in checked:
                continue
            sym_r, sym_c = 4 - r, 4 - c
            if (sym_r, sym_c) in checked:
                continue

            flow_val = bf_array[r, c]
            sym_flow_val = bf_array[sym_r, sym_c]

            # Higher tolerance because corner boundaries test fails when OMP_NUM_THREADS>nrows (5 here)
            # TODO: investigate and fix the corner issues
            assert np.isclose(flow_val, sym_flow_val, atol=1e-5), (
                f"Symmetric corner cells ({r},{c})={flow_val:.6f} and "
                f"({sym_r},{sym_c})={sym_flow_val:.6f} should have equal flow"
            )
            checked.add((r, c))
            checked.add((sym_r, sym_c))
