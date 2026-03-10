"""
Memory-based tests for 5x5 domain simulations.

These tests replace GRASS-based tests from tests/grass_tests/test_itzi.py
(except test_region_mask which remains GRASS-specific).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from collections import namedtuple
from typing import Dict, List, TYPE_CHECKING

import numpy as np
import pytest

from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.providers.domain_data import DomainData
from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider
from itzi.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
from itzi.const import InfiltrationModelType, TemporalType

if TYPE_CHECKING:
    from itzi.simulation import Simulation

# =============================================================================
# Helper Functions for Building SimulationConfig
# =============================================================================


def make_input_map_names(**overrides) -> Dict[str, str | None]:
    """Generate default input_map_names dict from ARRAY_DEFINITIONS.

    Keys set to None are inactive; keys set to a truthy string activate features.
    """
    names = {ad.key: None for ad in ARRAY_DEFINITIONS if ArrayCategory.INPUT in ad.category}
    names.update(overrides)
    return names


def make_output_map_names(prefix: str, keys: List[str]) -> Dict[str, str | None]:
    """Generate output_map_names dict from ARRAY_DEFINITIONS.

    Args:
        prefix: Prefix for output map names (e.g., "out_5by5")
        keys: List of output keys to activate

    Returns:
        Dict with all OUTPUT-category keys, activated ones set to "prefix_keyname"
    """
    names = {ad.key: None for ad in ARRAY_DEFINITIONS if ArrayCategory.OUTPUT in ad.category}
    for k in keys:
        names[k] = f"{prefix}_{k}"
    return names


# =============================================================================
# Domain Data Container
# =============================================================================


Domain5by5Data = namedtuple(
    "Domain5by5Data",
    [
        "domain_data",  # DomainData instance
        "arr_dem_flat",  # DEM with z=0
        "arr_dem_high",  # DEM with z=132
        "arr_n",  # Manning's n = 0.05
        "arr_start_h",  # Initial depth: 0.2 at center [2,2]
        "arr_start_wse",  # Initial WSE: 132.2 at center [2,2]
        "arr_mask",  # All False (no mask)
        "arr_bctype",  # Boundary condition type for open boundaries
        "arr_rain",  # Rainfall in m/s
        "arr_inf",  # Infiltration in m/s
        "arr_loss",  # Losses in m/s
        "arr_inflow",  # Inflow in m/s
    ],
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def domain_5by5() -> Domain5by5Data:
    """Create a 5x5 domain with all base arrays.

    This fixture provides the foundational data for all 5x5 tests:
    - 5x5 grid at 10m resolution
    - Domain extends: north=50, south=0, east=50, west=0
    - Total area: 2500 m²
    """
    # Domain dimensions
    rows, cols = 5, 5
    north, south, east, west = 50.0, 0.0, 50.0, 0.0

    # Create DomainData
    domain_data = DomainData(
        north=north, south=south, east=east, west=west, rows=rows, cols=cols, crs_wkt=""
    )

    # DEM arrays
    arr_dem_flat = np.zeros(domain_data.shape, dtype=np.float32)
    arr_dem_high = np.full(domain_data.shape, 132.0, dtype=np.float32)

    # Manning's n
    arr_n = np.full(domain_data.shape, 0.05, dtype=np.float32)

    # Initial water depth: 0.2m at center cell [2, 2], 0 elsewhere
    arr_start_h = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_h[2, 2] = 0.2

    # Initial water surface elevation: 132.2m at center cell [2, 2]
    # (high DEM + 0.2m depth)
    arr_start_wse = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_wse[2, 2] = 132.2

    # No mask - whole domain active
    arr_mask = np.full(domain_data.shape, False, dtype=np.bool_)

    # Boundary condition type: 2 (open) at all 16 edge cells
    arr_bctype = np.zeros(domain_data.shape, dtype=np.float32)
    # Top and bottom rows
    arr_bctype[0, :] = 2
    arr_bctype[4, :] = 2
    # Left and right columns (excluding corners already set)
    arr_bctype[:, 0] = 2
    arr_bctype[:, 4] = 2

    # Rate arrays in m/s
    # Rainfall: 10 mm/h = 10/(1000*3600) m/s
    arr_rain = np.full(domain_data.shape, 10.0 / (1000 * 3600), dtype=np.float32)
    # Infiltration: 2 mm/h = 2/(1000*3600) m/s
    arr_inf = np.full(domain_data.shape, 2.0 / (1000 * 3600), dtype=np.float32)
    # Losses: 1.5 mm/h = 1.5/(1000*3600) m/s
    arr_loss = np.full(domain_data.shape, 1.5 / (1000 * 3600), dtype=np.float32)
    # Inflow: 0.1 m/s (already in m/s)
    arr_inflow = np.full(domain_data.shape, 0.1, dtype=np.float32)

    return Domain5by5Data(
        domain_data=domain_data,
        arr_dem_flat=arr_dem_flat,
        arr_dem_high=arr_dem_high,
        arr_n=arr_n,
        arr_start_h=arr_start_h,
        arr_start_wse=arr_start_wse,
        arr_mask=arr_mask,
        arr_bctype=arr_bctype,
        arr_rain=arr_rain,
        arr_inf=arr_inf,
        arr_loss=arr_loss,
        arr_inflow=arr_inflow,
    )


@pytest.fixture(scope="module")
def sim_5by5(domain_5by5: Domain5by5Data) -> Simulation:
    """Run a 5x5 simulation for 60s with 30s record step.

    Outputs: water_depth, water_surface_elevation, froude, v, vdir, qx, qy, volume_error
    """
    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 1, 0),  # 60 seconds
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
        ),
        output_map_names=make_output_map_names(
            "out_5by5",
            [
                "water_depth",
                "water_surface_elevation",
                "froude",
                "v",
                "vdir",
                "qx",
                "qy",
                "volume_error",
            ],
        ),
        # Same values as original 5by5.ini
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
        infiltration_model=InfiltrationModelType.NULL,
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

    # Run simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return simulation


# =============================================================================
# Tests
# =============================================================================


class TestNumberOfOutput:
    """Test that the correct number of output maps are produced.
    60 seconds test with 30s record steps.
    Memory provider outputs: initial (t=0) + 2 record steps (t=30, t=60) = 3 outputs.
    Note: Unlike GRASS provider, memory provider does not generate separate max maps."""

    def test_water_depth_count(self, sim_5by5):
        """water_depth should have 3 outputs (initial + 2 record steps)."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["water_depth"]) == 3

    def test_water_surface_elevation_count(self, sim_5by5):
        """water_surface_elevation should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["water_surface_elevation"]) == 3

    def test_froude_count(self, sim_5by5):
        """froude should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["froude"]) == 3

    def test_v_count(self, sim_5by5):
        """v (velocity) should have 3 outputs (initial + 2 record steps)."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["v"]) == 3

    def test_vdir_count(self, sim_5by5):
        """vdir (velocity direction) should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["vdir"]) == 3

    def test_qx_count(self, sim_5by5):
        """qx should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["qx"]) == 3

    def test_qy_count(self, sim_5by5):
        """qy should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["qy"]) == 3

    def test_volume_error_count(self, sim_5by5):
        """volume_error should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["volume_error"]) == 3
