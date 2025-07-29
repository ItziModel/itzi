# coding=utf8
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

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from collections import Counter

import numpy as np


class ArrayCategory(Enum):
    """Categories of arrays in the simulation"""

    INPUT = "INPUT"  # Read from external sources (maps, time series)
    INTERNAL = "INTERNAL"  # Computed during simulation (state variables)
    ACCUMULATION = "ACCUMULATION"  # Time-integrated values for reporting
    OUTPUT = "OUTPUT"  # Derived arrays for output/reporting


@dataclass
class ArrayDefinition:
    """Complete definition of a simulation array"""

    key: str  # identifier
    # https://csdms.colorado.edu/wiki/CSN_Searchable_List
    csdms_name: str  # For BMI
    # https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
    cf_name: str  # For CF compliant NetCDF
    category: list[ArrayCategory]  # Array category
    description: str  # Human-readable description
    unit: str  # Physical units of the array
    cf_unit: str  # The unit expected by the CF convention
    var_loc: str  # Location of the value. Either "face" or "edge"
    fill_value: float = 0.0  # Fill value (replace NaN)
    computes_from: Optional[str] = None  # For accumulation arrays


# Centralized array definitions - Single source of truth
_INPUT_ARRAY_DEFINITIONS = [
    # ===== INPUT ARRAYS =====
    # These arrays are read from external sources (maps, time series)
    ArrayDefinition(
        key="dem",
        csdms_name="land_surface__elevation",
        cf_name="ground_level_altitude",
        category=[ArrayCategory.INPUT],
        description="Digital Elevation Model.",
        unit="m",
        cf_unit="m",
        var_loc="face",
        fill_value=np.finfo(np.float32).max,
    ),
    ArrayDefinition(
        key="friction",
        csdms_name="land_surface_water_flow__manning_n_parameter",
        cf_name="",
        category=[ArrayCategory.INPUT],
        description="Manning's n value.",
        unit="s m(-1/3)",
        cf_unit="",
        var_loc="face",
        fill_value=1.0,
    ),
    ArrayDefinition(
        key="water_depth",
        csdms_name="land_surface_water__depth",
        cf_name="flood_water_thickness",
        category=[ArrayCategory.INPUT, ArrayCategory.OUTPUT],
        description="Water depth.",
        unit="m",
        cf_unit="m",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="effective_porosity",
        csdms_name="soil_water__effective_porosity",
        cf_name="soil_porosity",
        category=[ArrayCategory.INPUT],
        description="Porosity available to contribute to fluid flow.",
        unit="m/m",
        cf_unit="m",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="capillary_pressure",
        csdms_name="soil_water__pressure_head",
        cf_name="soil_suction_at_saturation",
        category=[ArrayCategory.INPUT],
        description="Soil capillary pressure. Also called suction head.",
        unit="m",
        cf_unit="Pa",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="hydraulic_conductivity",
        csdms_name="soil_water__hydraulic_conductivity",
        cf_name="soil_hydraulic_conductivity_at_saturation",
        category=[ArrayCategory.INPUT],
        description="Soil’s ability to transmit water through its pores during infiltration.",
        unit="m s-1",
        cf_unit="m s-1",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="soil_water_content",
        csdms_name="soil_water__volume_fraction",
        cf_name="soil_liquid_water_content",
        category=[ArrayCategory.INPUT],
        description="Relative soil water content.",
        unit="m/m",
        cf_unit="kg m-2",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="infiltration",
        csdms_name="soil_surface_water__infiltration_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.INPUT],
        description="User-defined infiltration rate.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="losses",
        csdms_name="land_surface_water__losses_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.INPUT],
        description="User-defined water losses.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="rain",
        csdms_name="atmosphere_water__precipitation_leq-volume_flux",
        cf_name="rainfall_rate",
        category=[ArrayCategory.INPUT],
        description="User-provided precipitation rate.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="inflow",
        csdms_name="land_surface_water__inflow_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.INPUT],
        description="User-defined inflow volume flux.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="bcval",
        csdms_name="land_surface_water__boundary_value",
        cf_name="",
        category=[ArrayCategory.INPUT],
        description="Boundary condition value.",
        unit="1",
        cf_unit="",
        var_loc="face",
        fill_value=0.0,
    ),
    ArrayDefinition(
        key="bctype",
        csdms_name="land_surface_water__boundary_type",
        cf_name="",
        category=[ArrayCategory.INPUT],
        description="Boundary condition type.",
        unit="1",
        cf_unit="",
        var_loc="face",
        fill_value=0.0,
    ),
]
# ===== INTERNAL ARRAYS =====
# These arrays are computed during simulation (state variables and intermediate calculations)
# Note: Many are also exported
_INTERNAL_ARRAY_DEFINITIONS = [
    ArrayDefinition(
        key="computed_infiltration",
        csdms_name="soil_surface_water__infiltration_volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Calculated instantaneous infiltration rate.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="eff_precip",
        csdms_name="land_surface_water__effective_precipitation_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Effective precipitation after adding/removing "
        "rainfall, infiltration, evapotranspiration, etc.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="hfe",
        csdms_name="land_surface_water__x_component_of_depth",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Water depth at eastern cell edge.",
        unit="m",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="hfs",
        csdms_name="land_surface_water__y_component_of_depth",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Water depth at southern cell edge.",
        unit="m",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="qe",
        csdms_name="land_surface_water__x_component_of_old_volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Water flux at eastern cell edge (previous timestep).",
        unit="m2 s-1",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="qs",
        csdms_name="land_surface_water__y_component_of_old_volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Water flux at southern cell edge (previous timestep).",
        unit="m2 s-1",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="qe_new",
        csdms_name="land_surface_water__x_component_of_volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Water flux at eastern cell edge.",
        unit="m2 s-1",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="qs_new",
        csdms_name="land_surface_water__y_component_of_volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Water flux at southern cell edge.",
        unit="m2 s-1",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="hmax",
        csdms_name="land_surface_water__max_of_depth",
        cf_name="",
        category=[ArrayCategory.INTERNAL],  # output together with depth arrays
        description="Maximum water depth reached since the beginning of the simulation.",
        unit="m",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="v",
        csdms_name="land_surface_water_flow__speed",
        cf_name="",
        category=[ArrayCategory.INTERNAL, ArrayCategory.OUTPUT],
        description="Overland flow speed (velocity’s magnitude).",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="vdir",
        csdms_name="land_surface_water_flow__azimuth_angle_of_velocity",
        cf_name="",
        category=[ArrayCategory.INTERNAL, ArrayCategory.OUTPUT],
        description="Velocity’s direction. Counter-clockwise from East.",
        unit="degree",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="vmax",
        csdms_name="land_surface_water_flow__max_of_speed",
        cf_name="",
        category=[ArrayCategory.INTERNAL],  # output together with speed arrays
        description="Maximum water speed reached since the beginning of the simulation.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="froude",
        csdms_name="land_surface_water_flow__froude_number",
        cf_name="",
        category=[ArrayCategory.INTERNAL, ArrayCategory.OUTPUT],
        description="Froude number: ratio of flow inertia to gravity",
        unit="1",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="n_drain",
        csdms_name="land_surface_water__drainage_network_inflow_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Inflow from the drainage network. Negative when water is leaving the raster domain",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="capped_losses",
        csdms_name="land_surface_water__capped_losses_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Losses capped to water depth to prevent negative depths.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="dire",
        csdms_name="land_surface_water__x_component_of_flow_direction",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Rain routing at the eastern cell edge "
        "0: the flow is going dowstream, index-wise, "
        "1: the flow is going upstream, index-wise "
        "-1: no routing happening on that face",
        unit="1",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="dirs",
        csdms_name="land_surface_water__y_component_of_flow_direction",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Rain routing at the southern cell edge "
        "0: the flow is going dowstream, index-wise, "
        "1: the flow is going upstream, index-wise "
        "-1: no routing happening on that face",
        unit="1",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="ext",
        csdms_name="land_surface_water__external_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.INTERNAL],
        description="Sum of inflow, n_drain, and eff_precip",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
    ),
]
# ===== ACCUMULATION ARRAYS =====
# These arrays accumulate volumes over a reporting interval
_ACCUM_ARRAY_DEFINITIONS = [
    ArrayDefinition(
        key="boundaries_accum",
        csdms_name="land_surface_water__time_integral_of_boundary_volume_flux",
        cf_name="",
        category=[ArrayCategory.ACCUMULATION],
        description="The total amount of water entering the domain due to boundary conditions. "
        "Negative if water is leaving the domain",
        unit="m",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="infiltration_accum",
        csdms_name="soil_surface_water__time_integral_of_infiltration_volume_flux",
        cf_name="",
        category=[ArrayCategory.ACCUMULATION],
        description="The total amount of water leaving the domain due to infiltration.",
        unit="m",
        cf_unit="",
        var_loc="face",
        computes_from="computed_infiltration",
    ),
    ArrayDefinition(
        key="rainfall_accum",
        csdms_name="atmosphere_water__time_integral_of_precipitation_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.ACCUMULATION],
        description="The total amount of water entering the domain due to rainfall.",
        unit="m",
        cf_unit="",
        var_loc="face",
        computes_from="rain",
    ),
    ArrayDefinition(
        key="inflow_accum",
        csdms_name="land_surface_water__time_integral_of_inflow_volume_flux",
        cf_name="",
        category=[ArrayCategory.ACCUMULATION],
        description="The total amount of water entering the domain due to user-defined inflow.",
        unit="m",
        cf_unit="",
        var_loc="face",
        computes_from="inflow",
    ),
    ArrayDefinition(
        key="losses_accum",
        csdms_name="land_surface_water__time_integral_of_capped_losses_volume_flux",
        cf_name="",
        category=[ArrayCategory.ACCUMULATION],
        description="The total amount of water leaving the domain due to user-defined losses.",
        unit="m",
        cf_unit="",
        var_loc="face",
        computes_from="capped_losses",
    ),
    ArrayDefinition(
        key="drainage_network_accum",
        csdms_name="land_surface_water__time_integral_of_drainage_network_inflow_volume_flux",
        cf_name="",
        category=[ArrayCategory.ACCUMULATION],
        description="The total amount of water entering the domain due to drainage network overflow. "
        "Negative if the water leaves teh domain due to inflow into the drainage network.",
        unit="m",
        cf_unit="",
        var_loc="face",
        computes_from="n_drain",
    ),
    ArrayDefinition(
        key="error_depth_accum",
        csdms_name="land_surface_water__time_integral_of_error_volume_flux",
        cf_name="",
        category=[ArrayCategory.ACCUMULATION],
        description="The total amount of water created due to numerical instabilities.",
        unit="m",
        cf_unit="",
        var_loc="face",
    ),
]
# ===== OUTPUT ARRAYS =====
# These arrays are calculated for reporting purposes only
_OUTPUT_ARRAY_DEFINITIONS = [
    ArrayDefinition(
        key="water_surface_elevation",
        csdms_name="land_surface_water__elevation",
        cf_name="",
        category=[ArrayCategory.INPUT, ArrayCategory.OUTPUT],
        description="Water surface elevation. Terrain elevation + water depth",
        unit="m",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="qx",
        csdms_name="land_surface_water__x_component_of_runoff_volume_flux",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Surface water volumetric flow at the eastern edge of the cell.",
        unit="m3 s-1",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="qy",
        csdms_name="land_surface_water__y_component_of_runoff_volume_flux",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Surface water volumetric flow at the southern edge of the cell.",
        unit="m3 s-1",
        cf_unit="",
        var_loc="edge",
    ),
    ArrayDefinition(
        key="volume_error",
        csdms_name="land_surface_water__time_integral_of_error_volume",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Cumulative volume of water created due to numerical instabilities.",
        unit="m3",
        cf_unit="",
        var_loc="face",
    ),
    ArrayDefinition(
        key="mean_boundary_flow",
        csdms_name="land_surface_water__time_mean_of_boundary_volume_flux",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Mean of volumetric flow passing through the domain boundaries since the last record. "
        "Negative if water is leaving the domain.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        computes_from="boundaries_accum",
    ),
    ArrayDefinition(
        key="mean_inflow",
        csdms_name="land_surface_water__time_mean_of_inflow_volume_flux",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Mean of volumetric flow entering the domain due to "
        "user-defined inflow volume flux since the last record.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        computes_from="inflow_accum",
    ),
    ArrayDefinition(
        key="mean_losses",
        csdms_name="land_surface_water__time_mean_of_losses_volume_flux",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Mean of volumetric flow leaving the domain due to "
        "user-defined losses since the last record.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        computes_from="losses_accum",
    ),
    ArrayDefinition(
        key="mean_drainage_flow",
        csdms_name="land_surface_water__time_mean_of_drainage_network_inflow_volume_flux",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Mean of volumetric flow entering the surface domain due to "
        "network drainage overflow since the last record."
        "Negative when the water leaves the surface domain due to inflow into the drainage network.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        computes_from="drainage_network_accum",
    ),
    ArrayDefinition(
        key="mean_infiltration",
        csdms_name="soil_surface_water__time_mean_of_infiltration_volume_flux",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Mean of infiltration rate.",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        computes_from="infiltration_accum",
    ),
    ArrayDefinition(
        key="mean_rainfall",
        csdms_name="atmosphere_water__time_mean_of_precipitation_leq-volume_flux",
        cf_name="",
        category=[ArrayCategory.OUTPUT],
        description="Mean of rainfall rate",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        computes_from="rainfall_accum",
    ),
]

ARRAY_DEFINITIONS = (
    _INPUT_ARRAY_DEFINITIONS
    + _INTERNAL_ARRAY_DEFINITIONS
    + _ACCUM_ARRAY_DEFINITIONS
    + _OUTPUT_ARRAY_DEFINITIONS
)

number_of_array_definitions = len(ARRAY_DEFINITIONS)

# Some sanity check
for attr in ["key", "csdms_name", "cf_name", "description"]:
    all_values = [getattr(arr_def, attr) for arr_def in ARRAY_DEFINITIONS]
    # No empty name
    if "" in all_values and not attr == "cf_name":
        raise ValueError(f"Found empty names in <{attr}>.")
    # Make sure there is no duplicates
    values_counts = Counter(all_values)
    duplicates = [item for item, count in values_counts.items() if count > 1]
    if 0 < len(duplicates):
        if attr == "cf_name" and len(duplicates) == 1 and "" in duplicates:
            continue
        raise ValueError(f"Found duplicates in <{attr}>: {duplicates}")
