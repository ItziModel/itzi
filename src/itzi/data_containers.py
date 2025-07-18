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

from typing import Dict, Tuple, ClassVar
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class DrainageLinkData:
    """Store the instantaneous state of a node during a drainage simulation"""

    vertices: Tuple[Tuple[float, float], ...]
    attributes: Tuple  # one values for each columns, minus "cat"
    columns_definition: ClassVar[Tuple[Tuple[str, str], ...]] = (
        ("cat", "INTEGER PRIMARY KEY"),
        ("link_id", "TEXT"),
        ("type", "TEXT"),
        ("flow", "REAL"),
        ("depth", "REAL"),
        #    (u'velocity', 'REAL'),
        ("volume", "REAL"),
        ("offset1", "REAL"),
        ("offset2", "REAL"),
        #    (u'yFull', 'REAL'),
        ("froude", "REAL"),
    )

    def __post_init__(self):
        """Validate attributes length after initialization."""
        expected_len = len(self.columns_definition) - 1
        if len(self.attributes) != expected_len:
            raise ValueError(
                f"DrainageLinkData: Incorrect number of attributes. "
                f"Expected {expected_len}, got {len(self.attributes)}"
            )


@dataclass(frozen=True)
class DrainageNodeData:
    """Store the instantaneous state of a node during a drainage simulation"""

    coordinates: Tuple[float, float]
    attributes: Tuple  # one values for each columns, minus "cat"
    columns_definition: ClassVar[Tuple[Tuple[str, str], ...]] = (
        ("cat", "INTEGER PRIMARY KEY"),
        ("node_id", "TEXT"),
        ("type", "TEXT"),
        ("linkage_type", "TEXT"),
        ("linkage_flow", "REAL"),
        ("inflow", "REAL"),
        ("outflow", "REAL"),
        ("latFlow", "REAL"),
        ("losses", "REAL"),
        ("overflow", "REAL"),
        ("depth", "REAL"),
        ("head", "REAL"),
        #    (u'crownElev', 'REAL'),
        ("crestElev", "REAL"),
        ("invertElev", "REAL"),
        ("initDepth", "REAL"),
        ("fullDepth", "REAL"),
        ("surDepth", "REAL"),
        ("pondedArea", "REAL"),
        #    (u'degree', 'INT'),
        ("newVolume", "REAL"),
        ("fullVolume", "REAL"),
    )

    def __post_init__(self):
        """Validate attributes length after initialization."""
        expected_len = len(self.columns_definition) - 1
        if len(self.attributes) != expected_len:
            raise ValueError(
                f"DrainageNodeData: Incorrect number of attributes. "
                f"Expected {expected_len}, got {len(self.attributes)}"
            )


@dataclass(frozen=True)
class DrainageNetworkData:
    nodes: Tuple[DrainageNodeData, ...]
    links: Tuple[DrainageLinkData, ...]


@dataclass(frozen=True)
class ContinuityData:
    """Store information about simulation continuity"""

    new_domain_vol: float
    volume_change: float
    volume_error: float
    continuity_error: float


@dataclass(frozen=True)
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
