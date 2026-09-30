"""
Copyright (C) 2015-2026 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import copy
import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import ClassVar, NamedTuple

import grass.pygrass.utils as gutils
import grass.script as gscript
import grass.temporal as tgis
import numpy as np
from grass.pygrass.gis.region import Region
from grass.pygrass.vector import VectorTopo
from grass.pygrass.vector.geometry import Line, Point
from grass.pygrass.vector.table import Link, Table
from itzi_core import DomainData, TemporalType
from itzi_core.data_containers import (
    DrainageLinkAttributes,
    DrainageNetworkAttributes,
    DrainageNetworkTopology,
    DrainageNodeAttributes,
)

import itzi.messenger as msgr
from itzi.ensemble.models import EffectiveMask
from itzi.grass.utils import (
    MapData,
    RasterMapType,
    STDSType,
    name_is_map,
    output_name,
    read_domain,
    replace_cell_null_sentinel,
    resolve_effective_mask,
    split_identifier,
    write_raster_map_blocking,
)


class DBLinkDescription(NamedTuple):
    layer: int
    table: Table


class DBLayerDescription(NamedTuple):
    table_suffix: str
    columns: tuple[tuple[str, str], ...]
    layer_number: int


class GrassInterface:
    """
    A class providing an access to GRASS GIS Python interfaces:
    scripting, pygrass, temporal GIS
    The interface of this class relies on numpy arrays for raster values.
    Everything related to GRASS maps or stds stays in that class.
    """

    # a unit convertion table relative to seconds
    time_unit_conv: ClassVar[Mapping[str, int]] = MappingProxyType(
        {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
    )
    # datatype conversion between GRASS and numpy
    dtype_conv: ClassVar[Mapping[str, tuple]] = MappingProxyType(
        {
            "FCELL": ("float16", "float32"),
            "DCELL": ("float_", "float64"),
            "CELL": (
                "bool_",
                "int_",
                "intc",
                "intp",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            ),
        }
    )

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        dtype,
        region_id: str | None,
        raster_mask_id: str | None,
        effective_mask: EffectiveMask | None = None,
    ) -> None:
        assert isinstance(start_time, datetime), "start_time not a datetime object!"
        assert isinstance(end_time, datetime), "end_time not a datetime object!"
        assert start_time <= end_time, "start_time > end_time!"

        self.region_id = region_id
        self.start_time = start_time
        self.end_time = end_time
        self.dtype = dtype

        # LatLon is not supported
        if gscript.locn_is_latlong():
            msgr.fatal("latlong location is not supported. Please use a projected location")
        # Set region
        if self.region_id:
            gscript.use_temp_region()
            gscript.run_command("g.region", region=region_id)
        self.region = Region()
        self.xr = self.region.cols
        self.yr = self.region.rows
        # Check if region is at least 3x3
        if self.xr < 3 or self.yr < 3:
            msgr.fatal("GRASS Region should be at least 3 cells by 3 cells")
        selected_mask = effective_mask or resolve_effective_mask(raster_mask_id)
        if selected_mask.mode != "none" and selected_mask.source is None:
            msgr.fatal(f"Effective {selected_mask.mode} mask has no source")
        self.mask_mode = selected_mask.mode
        self.mask_source = selected_mask.source if selected_mask.mode != "none" else None
        self.overwrite = gscript.overwrite()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def __del__(self):
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the temporary region."""
        if self.region_id:
            msgr.debug("Remove temp region...")
            gscript.del_temp_region()

    def grass_dtype(self, dtype: str) -> RasterMapType:
        if dtype in self.dtype_conv["DCELL"]:
            mtype = "DCELL"
        elif dtype in self.dtype_conv["CELL"]:
            mtype = "CELL"
        elif dtype in self.dtype_conv["FCELL"]:
            mtype = "FCELL"
        else:
            raise ValueError("datatype incompatible with GRASS!")
        return mtype

    def to_s(self, unit: str, time: int) -> int:
        """Change an input time into seconds"""
        assert isinstance(unit, str), f"{unit} Not a string"
        return self.time_unit_conv[unit] * time

    def from_s(self, unit: str, time: float) -> int:
        """Change an input time from seconds to another unit"""
        assert isinstance(unit, str), f"{unit} Not a string"
        return int(time / self.time_unit_conv[unit])

    def to_datetime(self, unit: str, time: int) -> datetime:
        """Take a number and a unit as entry
        return a datetime object relative to start_time
        usefull for assigning start_time and end_time
        to maps from relative stds
        """
        return self.start_time + timedelta(seconds=self.to_s(unit, time))

    def get_domain_data(self) -> DomainData:
        return read_domain(None)

    def get_npmask(self) -> np.ndarray:
        """Return a boolean numpy ndarray where True is outside the domain."""
        if self.mask_mode == "none":
            return np.full(shape=(self.yr, self.xr), fill_value=False, dtype=np.bool_)
        assert self.mask_source is not None
        grass_mask = self.read_raster_map(self.mask_source)
        if self.mask_mode == "explicit":
            # r.mask accepts every non-NULL cell, including a zero value.
            return np.isnan(grass_mask)
        # A mapset MASK uses CELL semantics: zero and NULL are outside.
        return np.isnan(grass_mask) | (grass_mask == 0)

    def get_sim_extend_in_stds_unit(self, strds) -> tuple[int | datetime, int | datetime]:
        """Take a strds object as input
        Return the simulation start_time and end_time, expressed in
        the unit of the input strds
        """
        if strds.get_temporal_type() == TemporalType.RELATIVE:
            # get start time and end time in seconds
            rel_end_time = (self.end_time - self.start_time).total_seconds()
            rel_unit = strds.get_relative_time_unit()
            if rel_unit not in self.time_unit_conv:
                supported_units = ", ".join(sorted(self.time_unit_conv))
                if rel_unit is None:
                    msgr.fatal(f"STRDS <{strds.get_id()}> has no relative time unit")
                msgr.fatal(
                    f"STRDS <{strds.get_id()}> uses unsupported relative time unit "
                    f"<{rel_unit}>; supported units are {supported_units}"
                )
            start_time_in_stds_unit = 0
            end_time_in_stds_unit = self.from_s(rel_unit, rel_end_time)
        elif strds.get_temporal_type() == TemporalType.ABSOLUTE:
            start_time_in_stds_unit = self.start_time
            end_time_in_stds_unit = self.end_time
        else:
            assert False, "unknown temporal type"
        return start_time_in_stds_unit, end_time_in_stds_unit

    def stds_temporal_sanity(
        self, stds_id: str, stds: tgis.SpaceTimeRasterDataset | None = None
    ) -> bool:
        """Make the following check on the given stds:
        - Topology is valid
        - No gap
        - Cover all simulation time
        return True if all the above is True, False otherwise
        """
        out = True
        if stds is None:
            stds = tgis.open_stds.open_old_stds(stds_id, "strds")
        stds_start, stds_end = stds.get_temporal_extent_as_tuple()
        if stds_start is None or stds_end is None:
            msgr.fatal(
                f"STRDS <{stds_id}> has no temporal extent; "
                "make sure it contains registered raster maps"
            )
        maps = stds.get_registered_maps_as_objects(order="start_time")
        # valid topology
        if not stds.check_temporal_topology(maps=maps):
            out = False
            msgr.warning(f"{stds_id}: invalid topology")
        # no gap
        if stds.count_gaps(maps=maps) != 0:
            out = False
            msgr.warning(f"{stds_id}: gaps found")
        # cover all simulation time
        sim_start, sim_end = self.get_sim_extend_in_stds_unit(stds)
        if stds_start > sim_start:
            out = False
            msgr.warning(f"{stds_id}: starts after simulation")
        if stds_end < sim_end:
            out = False
            msgr.warning(f"{stds_id}: ends before simulation")
        return out

    def raster_list_from_strds(self, strds_name: str) -> list[MapData]:
        """Return a list of maps from a given strds
        for all the simulation duration
        Each map data is stored as a MapData namedtuple
        """
        assert isinstance(strds_name, str), "expect a string"

        # transform simulation start and end time in strds unit
        strds = tgis.open_stds.open_old_stds(strds_name, "strds")
        if not self.stds_temporal_sanity(strds_name, strds):
            msgr.fatal(f"{strds_name}: inadequate temporal format")
        sim_start, sim_end = self.get_sim_extend_in_stds_unit(strds)

        # retrieve data from DB
        where = f"start_time <= '{sim_end!s}' AND end_time >= '{sim_start!s}'"
        maplist = strds.get_registered_maps(
            columns=",".join(MapData._fields), where=where, order="start_time"
        )
        # check if every map exist
        maps_not_found = [m[0] for m in maplist if not name_is_map(m[0])]
        if any(maps_not_found):
            err_msg = "STRDS <{}>: Can't find following maps: {}"
            str_lst = ",".join(maps_not_found)
            msgr.fatal(err_msg.format(strds_name, str_lst))
        # change time data to datetime format
        if strds.get_temporal_type() == TemporalType.RELATIVE:
            rel_unit = strds.get_relative_time_unit()
            maplist = [
                (
                    i[0],
                    self.to_datetime(rel_unit, i[1]),
                    self.to_datetime(rel_unit, i[2]),
                )
                for i in maplist
            ]
        return [MapData(*i) for i in maplist]

    def validate_output_stds_temporal_type(
        self,
        stds_id: str,
        stds_type: STDSType,
        expected_temporal_type: TemporalType,
    ) -> None:
        """Fail early when overwriting an STDS with incompatible temporal type."""
        if not self.overwrite:
            return

        output_name(stds_id)
        if not tgis.dataset_factory(stds_type, stds_id).is_in_db():
            return
        existing_stds = tgis.open_stds.open_old_stds(stds_id, stds_type)
        existing_temporal_type = TemporalType(existing_stds.get_temporal_type())
        if existing_temporal_type != expected_temporal_type:
            msgr.fatal(
                f"Output {stds_type.upper()} <{stds_id}> already exists with "
                f"{existing_temporal_type} temporal type and cannot be overwritten "
                f"by a {expected_temporal_type} simulation"
            )

    def read_raster_map(self, raster_id: str) -> np.ndarray:
        """Read a raster through GRASS 8.4's no-mask row APIs.

        The active mapset MASK must not alter model input values.  The one
        effective mask selected for the simulation is applied separately by
        :meth:`get_npmask` and passed to itzi-core.
        """
        from grass.lib import raster as libraster

        name, mapset = split_identifier(raster_id)
        if not mapset:
            msgr.fatal(f"Raster <{raster_id}> not found")

        raster_type = libraster.Rast_map_type(name, mapset)
        reader_data = {
            libraster.CELL_TYPE: (libraster.Rast_allocate_c_buf, libraster.Rast_get_c_row_nomask),
            libraster.FCELL_TYPE: (libraster.Rast_allocate_f_buf, libraster.Rast_get_f_row_nomask),
            libraster.DCELL_TYPE: (libraster.Rast_allocate_d_buf, libraster.Rast_get_d_row_nomask),
        }.get(raster_type)
        if reader_data is None:
            msgr.fatal(f"Raster <{raster_id}> has an unsupported data type")
        allocate, read_row = reader_data
        file_descriptor = libraster.Rast_open_old(name, mapset)
        if file_descriptor < 0:
            msgr.fatal(f"Raster <{raster_id}> cannot be opened")
        buffer = allocate()
        try:
            array = np.empty((self.yr, self.xr), dtype=self.dtype)
            for row_index in range(self.yr):
                read_row(file_descriptor, buffer, row_index)
                array[row_index] = np.ctypeslib.as_array(buffer, shape=(self.xr,))
        finally:
            libraster.Rast_close(file_descriptor)
        return replace_cell_null_sentinel(
            {
                libraster.CELL_TYPE: "CELL",
                libraster.FCELL_TYPE: "FCELL",
                libraster.DCELL_TYPE: "DCELL",
            }[raster_type],
            array,
        )

    def write_raster_map(self, arr: np.ndarray, raster_id: str, mkey: str, hmin: float) -> None:
        """Take a numpy array and write it to GRASS DB"""
        write_raster_map_blocking(
            arr,
            output_name(raster_id),
            self.grass_dtype(str(arr.dtype)),
            mkey,
            hmin,
            self.overwrite,
        )

    def create_db_links(
        self, vect_map: VectorTopo, linking_elem: dict[str, DBLayerDescription]
    ) -> dict[str, DBLinkDescription]:
        """vect_map an open vector map"""
        dblinks = {}
        for layer_name, layer_dscr in linking_elem.items():
            # Create DB links
            dblink = Link(
                layer=layer_dscr.layer_number,
                name=layer_name,
                table=vect_map.name + layer_dscr.table_suffix,
                key="cat",
            )
            # add link to vector map
            if dblink not in vect_map.dblinks:
                vect_map.dblinks.add(dblink)
            # create table
            dbtable = dblink.table()
            dbtable.create(layer_dscr.columns, overwrite=self.overwrite)
            dblinks[layer_name] = DBLinkDescription(dblink.layer, dbtable)
        return dblinks

    def write_vector_map(
        self,
        topology: DrainageNetworkTopology,
        attributes: DrainageNetworkAttributes,
        map_id: str,
    ) -> None:
        """Write a vector map to GRASS GIS"""
        map_name = output_name(map_id)
        node_attributes = {node.node_id: node for node in attributes.nodes}
        link_attributes = {link.link_id: link for link in attributes.links}
        topology_node_ids = {node.node_id for node in topology.nodes}
        topology_link_ids = {link.link_id for link in topology.links}
        if topology_node_ids != set(node_attributes):
            raise ValueError("Drainage topology and attributes have different node IDs")
        if topology_link_ids != set(link_attributes):
            raise ValueError("Drainage topology and attributes have different link IDs")
        linking_elements = {
            "node": DBLayerDescription(
                table_suffix="_node",
                columns=DrainageNodeAttributes.get_columns_definition(),
                layer_number=1,
            ),
            "link": DBLayerDescription(
                table_suffix="_link",
                columns=DrainageLinkAttributes.get_columns_definition(),
                layer_number=2,
            ),
        }
        # set category manually
        cat_num = 1
        with VectorTopo(
            map_name, mapset=gutils.getenv("MAPSET"), mode="w", overwrite=self.overwrite
        ) as vector_map:
            # create db links and tables
            dblinks = self.create_db_links(vector_map, linking_elements)
            # dict to keep DB infos to write DB after geometries
            db_info = {k: [] for k in linking_elements}

            ## Write the points ##
            # Write in the correct layer
            map_layer, _ = dblinks["node"]
            vector_map.layer = map_layer
            for node in topology.nodes:
                if node.coordinates is not None:
                    point = Point(*node.coordinates)
                    # The write function of the vector map set the layer to the one we set earlier
                    vector_map.write(point, cat=cat_num)
                # Get DB attributes even if no associated geometry
                node_values = tuple(node_attributes[node.node_id].model_dump().values())
                attrs = (cat_num,) + node_values
                db_info["node"].append(attrs)
                cat_num += 1

            ## Write the lines ##
            # Set the vector map to the correct layer
            map_layer, _ = dblinks["link"]
            vector_map.layer = map_layer
            for link in topology.links:
                # assemble geometry
                if link.vertices and all(vertex is not None for vertex in link.vertices):
                    line_object = Line(link.vertices)
                    # The write function of the vector map set the layer to the one we set earlier
                    vector_map.write(line_object, cat=cat_num)
                # Get DB attributes even if no associated geometry
                link_values = tuple(link_attributes[link.link_id].model_dump().values())
                attrs = (cat_num,) + link_values
                db_info["link"].append(attrs)
                cat_num += 1

        # write attributes to DB
        for geom_type, attrs in db_info.items():
            map_layer, dbtable = dblinks[geom_type]
            for attr in attrs:
                dbtable.insert(attr)
            dbtable.conn.commit()

    def register_maps_in_stds(
        self,
        stds_title: str,
        stds_id: str,
        map_list: list[tuple[str, datetime | timedelta]],
        stds_type: STDSType,
        t_type: TemporalType,
    ) -> None:
        """Create a STDS, create one mapdataset for each map and
        register them in the temporal database
        """
        assert isinstance(stds_title, str), "not a string!"
        output_name(stds_id)
        assert isinstance(t_type, str), "not a string!"
        # Print message in case of decreased GRASS verbosity
        if msgr.verbosity() <= 2:
            msgr.message("Registering maps in temporal framework...")
        # Set GRASS verbosity to -1 to avoid superfluous messages about semantic labels
        grass_verbosity = copy.deepcopy(os.environ["GRASS_VERBOSE"])
        if stds_type == "stvds" and grass_verbosity != "-1":
            os.environ["GRASS_VERBOSE"] = "-1"
        # create stds
        stds_desc = ""
        stds = tgis.open_new_stds(
            stds_id,
            stds_type,
            t_type,
            stds_title,
            stds_desc,
            "mean",
            overwrite=self.overwrite,
        )

        dataset_type = {"strds": tgis.RasterDataset, "stvds": tgis.VectorDataset}[stds_type]
        map_dts_lst = []
        for map_id, map_time in map_list:
            output_name(map_id)
            map_dts = dataset_type(map_id)
            # load spatial data from map
            map_dts.load()
            # set time
            if t_type == TemporalType.RELATIVE:
                assert isinstance(map_time, timedelta)
                rel_time = map_time.total_seconds()
                map_dts.set_relative_time(rel_time, None, "seconds")
            elif t_type == TemporalType.ABSOLUTE:
                assert isinstance(map_time, datetime)
                map_dts.set_absolute_time(start_time=map_time)
            map_dts_lst.append(map_dts)
        # Finally register the maps
        t_unit = {TemporalType.RELATIVE: "seconds", TemporalType.ABSOLUTE: ""}
        stds_corresp = {"strds": "raster", "stvds": "vector"}
        del_empty = {"strds": True, "stvds": False}
        tgis.register.register_map_object_list(
            stds_corresp[stds_type],
            map_dts_lst,
            stds,
            delete_empty=del_empty[stds_type],
            unit=t_unit[t_type],
        )
        # Restore GRASS verbosity
        if stds_type == "stvds" and grass_verbosity != "-1":
            os.environ["GRASS_VERBOSE"] = grass_verbosity
