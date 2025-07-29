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

from typing import Dict, Tuple, TYPE_CHECKING
import dataclasses
from datetime import datetime

import numpy as np

if TYPE_CHECKING:
    from itzi.drainage import DrainageNode


@dataclasses.dataclass(frozen=True)
class DrainageNodeCouplingData:
    """Store the translation between coordinates and array location for a given drainage node."""

    node_id: str  # Name of the drainage node
    node_object: "DrainageNode"
    # Location in the coordinate system
    x: float | None
    y: float | None
    # Location in the array
    row: int | None
    col: int | None


@dataclasses.dataclass(frozen=True)
class DrainageAttributes:
    """A base class for drainage data attributes."""

    def get_columns_definition(self) -> list[tuple[str, str]]:
        """Return a list of tuples to create DB columns"""
        type_mapping = {str: "TEXT", int: "INT", float: "REAL"}
        db_columns_def = [("cat", "INTEGER PRIMARY KEY")]
        for f in dataclasses.fields(self):
            db_field = (f.name, type_mapping[f.type])
            db_columns_def.append(db_field)
        return db_columns_def


@dataclasses.dataclass(frozen=True)
class DrainageLinkAttributes(DrainageAttributes):
    link_id: str
    link_type: str
    flow: float
    depth: float
    volume: float
    inlet_offset: float
    outlet_offset: float
    froude: float


@dataclasses.dataclass(frozen=True)
class DrainageLinkData:
    """Store the instantaneous state of a node during a drainage simulation"""

    vertices: Tuple[Tuple[float, float], ...]
    attributes: DrainageLinkAttributes


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class DrainageNodeData:
    """Store the instantaneous state of a node during a drainage simulation"""

    coordinates: Tuple[float, float]
    attributes: DrainageNodeAttributes


@dataclasses.dataclass(frozen=True)
class DrainageNetworkData:
    nodes: Tuple[DrainageNodeData, ...]
    links: Tuple[DrainageLinkData, ...]


@dataclasses.dataclass(frozen=True)
class ContinuityData:
    """Store information about simulation continuity"""

    new_domain_vol: float
    volume_change: float
    volume_error: float
    continuity_error: float


@dataclasses.dataclass(frozen=True)
class SimulationData:
    """Immutable data container for passing raw simulation state to Report.

    This is a pure data structure containing only the "raw ingredients"
    needed for a report. All report-specific calculations (e.g., WSE,
    average rates) are performed by the Report class itself.
    """

    sim_time: datetime
    time_step: float  # time step duration
    time_steps_counter: int  # number of time steps since last update
    continuity_data: ContinuityData
    raw_arrays: Dict[str, np.ndarray]
    accumulation_arrays: Dict[str, np.ndarray]
    cell_dx: float  # cell size in east-west direction
    cell_dy: float  # cell size in north-south direction
    drainage_network_data: DrainageNetworkData | None


@dataclasses.dataclass(frozen=True)
class MassBalanceData:
    """Contains the fields written to the mass balance file"""

    simulation_time: datetime
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
