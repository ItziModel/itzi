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

from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from itzi.const import DefaultValues, TemporalType, InfiltrationModelType

if TYPE_CHECKING:
    from itzi.drainage import DrainageNode


class DrainageNodeCouplingData(BaseModel):
    """Store the translation between coordinates and array location for a given drainage node."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    node_id: str  # Name of the drainage node
    node_object: "DrainageNode"
    # Location in the coordinate system
    x: float | None
    y: float | None
    # Location in the array
    row: int | None
    col: int | None


class DrainageAttributes(BaseModel):
    """A base class for drainage data attributes."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def get_columns_definition(cls, cat_primary_key=True) -> list[tuple[str, str]]:
        """Return a list of tuples to create DB columns"""
        type_mapping = {str: "TEXT", int: "INT", float: "REAL"}
        db_columns_def = [("cat", "INTEGER PRIMARY KEY")]
        if not cat_primary_key:
            db_columns_def = []
        for field_name, field_info in cls.model_fields.items():
            db_field = (field_name, type_mapping[field_info.annotation])
            db_columns_def.append(db_field)
        return db_columns_def


class DrainageLinkAttributes(DrainageAttributes):
    link_id: str
    link_type: str
    flow: float
    depth: float
    volume: float
    inlet_offset: float
    outlet_offset: float
    froude: float


class DrainageLinkData(BaseModel):
    """Store the instantaneous state of a node during a drainage simulation.
    Vertices include the coordinates of the start and end nodes."""

    model_config = ConfigDict(frozen=True)

    vertices: None | Tuple[Tuple[float, float] | None, ...]
    attributes: DrainageLinkAttributes


class DrainageNodeAttributes(DrainageAttributes):
    node_id: str
    node_type: str
    coupling_type: str
    coupling_flow: float
    inflow: float
    outflow: float
    lateral_inflow: float
    losses: float
    overflow: float
    depth: float
    head: float
    # crownElev: float
    crest_elevation: float
    invert_elevation: float
    initial_depth: float
    full_depth: float
    surcharge_depth: float
    ponding_area: float
    # degree: int
    volume: float
    full_volume: float


class DrainageNodeData(BaseModel):
    """Store the instantaneous state of a node during a drainage simulation"""

    model_config = ConfigDict(frozen=True)

    coordinates: None | Tuple[float, float]
    attributes: DrainageNodeAttributes


class DrainageNetworkData(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: Tuple[DrainageNodeData, ...]
    links: Tuple[DrainageLinkData, ...]


class ContinuityData(BaseModel):
    """Store information about simulation continuity"""

    model_config = ConfigDict(frozen=True)

    new_domain_vol: float
    volume_change: float
    volume_error: float
    continuity_error: float


class SimulationData(BaseModel):
    """Immutable data container for passing raw simulation state to Report.

    This is a pure data structure containing only the "raw ingredients"
    needed for a report. All report-specific calculations (e.g., WSE,
    average rates) are performed by the Report class itself.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    sim_time: datetime
    time_step: float  # time step duration
    time_steps_counter: int  # number of time steps since last update
    continuity_data: ContinuityData | None  # Made optional for use in tests
    raw_arrays: Dict[str, np.ndarray]
    accumulation_arrays: Dict[str, np.ndarray]
    cell_dx: float  # cell size in east-west direction
    cell_dy: float  # cell size in north-south direction
    drainage_network_data: DrainageNetworkData | None


class MassBalanceData(BaseModel):
    """Contains the fields written to the mass balance file"""

    model_config = ConfigDict(frozen=True)

    simulation_time: datetime | timedelta
    average_timestep: float
    timesteps: int
    boundary_volume: float
    rainfall_volume: float
    infiltration_volume: float
    inflow_volume: float
    losses_volume: float
    drainage_network_volume: float
    domain_volume: float
    volume_change: float
    volume_error: float
    percent_error: float


class SurfaceFlowParameters(BaseModel):
    """Parameters for the surface flow model."""

    model_config = ConfigDict(frozen=True)

    hmin: float = DefaultValues.HFMIN
    cfl: float = DefaultValues.CFL
    theta: float = DefaultValues.THETA
    g: float = DefaultValues.G
    vrouting: float = DefaultValues.VROUTING
    dtmax: float = DefaultValues.DTMAX
    slope_threshold: float = DefaultValues.SLOPE_THRESHOLD
    max_slope: float = DefaultValues.MAX_SLOPE
    max_error: float = DefaultValues.MAX_ERROR


class SimulationConfig(BaseModel):
    """Configuration data for a simulation run."""

    model_config = ConfigDict(frozen=True)

    # Simulation times
    start_time: datetime
    end_time: datetime
    record_step: timedelta
    temporal_type: TemporalType
    # Input and output raster maps
    input_map_names: Dict[str, str | None]
    output_map_names: Dict[str, str | None]
    # Surface flow parameters
    surface_flow_parameters: SurfaceFlowParameters
    # Mass balance file
    stats_file: str | Path | None = None
    # Hydrology parameters
    dtinf: float = DefaultValues.DTINF
    infiltration_model: InfiltrationModelType = InfiltrationModelType.NULL
    # Drainage parameters
    swmm_inp: str | None = None
    drainage_output: str | None = None
    orifice_coeff: float = DefaultValues.ORIFICE_COEFF
    free_weir_coeff: float = DefaultValues.FREE_WEIR_COEFF
    submerged_weir_coeff: float = DefaultValues.SUBMERGED_WEIR_COEFF
