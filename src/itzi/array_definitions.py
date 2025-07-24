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

import numpy as np


class ArrayCategory(Enum):
    """Categories of arrays in the simulation"""

    INPUT = "INPUT"  # Read from external sources (maps, time series)
    INTERNAL = "INTERNAL"  # Computed during simulation (state variables)
    ACCUMULATION = "ACCUMULATION"  # Time-integrated values for reporting
    OUTPUT = "OUTPUT"  # Derived arrays for output/reporting only


@dataclass
class ArrayDefinition:
    """Complete definition of a simulation array"""

    key: str  # Internal identifier
    user_name: str  # User-facing name
    # https://csdms.colorado.edu/wiki/CSN_Searchable_List
    csdms_name: str  # For BMI
    # https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
    cf_name: str  # For CF compliant NetCDF
    category: ArrayCategory  # Array category
    description: str  # Human-readable description
    unit: str  # Physical units of the array
    cf_unit: str  # The unit expected by the CF convention
    var_loc: str  # Location of the value. Either "face" or "edge"
    fill_value: float = 0.0  # Fill value (replace NaN)
    accumulates_from: Optional[str] = None  # For accumulation arrays


# Centralized array definitions - Single source of truth
ARRAY_DEFINITIONS = [
    # ===== INPUT ARRAYS =====
    # These arrays are read from external sources (maps, time series)
    ArrayDefinition(
        key="dem",
        user_name="dem",
        csdms_name="land_surface__elevation",
        cf_name="ground_level_altitude",
        category=ArrayCategory.INPUT,
        description="Digital Elevation Model",
        unit="m",
        cf_unit="m",
        var_loc="face",
        fill_value=np.finfo(np.float32).max,
    ),
    ArrayDefinition(
        key="friction",
        user_name="friction",
        csdms_name="land_surface_water_flow__manning_n_parameter",
        cf_name="",
        category=ArrayCategory.INPUT,
        description="Manning's n value",
        unit="s m(-1/3)",
        cf_unit="",
        var_loc="face",
        fill_value=1.0,
    ),
    ArrayDefinition(
        user_name="start_h",
        internal_name="h",
        csdms_name="land_surface_water__depth",
        cf_name="flood_water_thickness",
        category=ArrayCategory.INPUT,
        description="Water depth",
        unit="m",
        cf_unit="m",
        var_loc="face",
        default_value=0.0,
    ),
    ArrayDefinition(
        user_name="effective_porosity",
        internal_name="effective_porosity",
        csdms_name="soil_water__effective_porosity",
        cf_name="soil_porosity",
        category=ArrayCategory.INPUT,
        description="Porosity available to contribute to fluid flow",
        unit="m/m",
        cf_unit="m",
        var_loc="face",
        default_value=0.0,
    ),
    ArrayDefinition(
        user_name="capillary_pressure",
        internal_name="capillary_pressure",
        csdms_name="soil_water__pressure_head",
        cf_name="soil_suction_at_saturation",
        category=ArrayCategory.INPUT,
        description="Soil capillary pressure. Also called suction head",
        unit="m",
        cf_unit="Pa",
        var_loc="face",
        default_value=0.0,
    ),
    ArrayDefinition(
        user_name="hydraulic_conductivity",
        internal_name="hydraulic_conductivity",
        csdms_name="soil_water__hydraulic_conductivity",
        cf_name="soil_hydraulic_conductivity_at_saturation",
        category=ArrayCategory.INPUT,
        description="",
        unit="m s-1",
        var_loc="face",
        default_value=0.0,
    ),
    ArrayDefinition(
        user_name="soil_water_content",
        internal_name="soil_water_content",
        csdms_name="soil_water__volume_fraction",
        cf_name="soil_liquid_water_content",
        category=ArrayCategory.INPUT,
        description="Relative soil water content",
        unit="m/m",
        cf_unit="kg m-2",
        var_loc="face",
        default_value=0.0,
    ),
    ArrayDefinition(
        user_name="infiltration",
        internal_name="in_inf",
        csdms_name="soil_surface_water__infiltration_volume_flux",
        cf_name="",
        category=ArrayCategory.INPUT,
        description="User-defined infiltration rate",
        unit="m s-1",
        cf_unit="",
        var_loc="face",
        default_value=0.0,
    ),
    "losses",
    "rain",
    "inflow",
    "bcval",
    "bctype",
    # ===== INTERNAL ARRAYS =====
    # These arrays are computed during simulation (state variables and intermediate calculations)
    # Note: 'h' is both an INPUT (initial condition) and INTERNAL (evolving state)
    # The same internal_name is used but it transitions from input to computed state
    "inf",
    "hmax",
    "ext",
    "y",
    "hfe",
    "hfs",
    "qe",
    "qs",
    "qe_new",
    "qs_new",
    "etp",
    "eff_precip",
    "ue",
    "us",
    "v",
    "vdir",
    "vmax",
    "froude",
    "n_drain",
    "capped_losses",
    "dire",
    "dirs",
    # ===== ACCUMULATION ARRAYS =====
    # These arrays accumulate volumes over a reporting interval
    "boundaries_accum",
    "infiltration_accum",
    "rainfall_accum",
    "etp_accum",
    "inflow_accum",
    "losses_accum",
    "drainage_network_accum",
    "error_depth_accum",
    # ===== OUTPUT ARRAYS =====
    # These arrays are calculated for reporting purposes only
    "wse",
    "qx",
    "qy",
    "verror",
]
