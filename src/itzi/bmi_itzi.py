#! /usr/bin/env python3
"""
Basic Model Interface implementation for the Itzï flood model.

Copyright (C) 2020-2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from datetime import timedelta

import numpy as np
from bmipy import Bmi
import itzi


class BmiItzi(Bmi):
    """A distributed dynamic flood model."""

    _name = "Itzï"
    _input_var_names = {
        "land_surface__elevation": "dem",
        "land_surface_water_flow__manning_n_parameter": "friction",
        "land_surface_water__depth": "h",
        "land_surface_water__precipitation_leq-volume_flux": "rain",
        "land_surface_water__inflow_volume_flux": "inflow",
        "land_surface_water__boundary_type": "bctype",
        "land_surface_water__boundary_value": "bcval",
        "soil_surface_water__infiltration_volume_flux": "in_inf",
        "soil_water__effective_porosity": "effective_porosity",
        "soil_water__pressure_head": "capillary_pressure",
        "soil_water__hydraulic_conductivity": "hydraulic_conductivity",
        "soil_water__volume_fraction": "soil_water_content",
        "land_surface_water__losses_volume_flux": "losses",
    }
    _output_var_names = {
        "land_surface_water__depth": "h",
        "land_surface_water_surface__elevation": "y",
        "land_surface_water_flow__speed": "v",
        "land_surface_water_flow__azimuth_angle_of_velocity": "vdir",
        "land_surface_water__x_component_of_volume_flow_rate": "qe_new",
        "land_surface_water__y_component_of_volume_flow_rate": "qs_new",
        "soil_surface_water__boundary_volume_flux": "st_bound",
        "soil_surface_water__mean_of_infiltration_volume_flux": "st_inf",
        "land_surface_water__mean_of_precipitation_leq-volume_flux": "st_rain",
        "land_surface_water__mean_of_inflow_volume_flux": "st_inflow",
        "land_surface_water__mean_of_losses_volume_flux": "st_losses",
        "land_surface_water__mean_of_drainage_volume_flux": "st_ndrain",
        "land_surface_water__time_integral_of_error_volume": "st_herr",
        "land_surface_water_flow__froude_number": "fr",
    }

    _var_names = {**_input_var_names, **_output_var_names}

    _var_units = {
        "land_surface__elevation": "m",
        "land_surface_water_flow__manning_n_parameter": "s m(-1/3)",
        "land_surface_water__precipitation_leq-volume_flux": "m s-1",
        "land_surface_water__inflow_volume_flux": "m s-1",
        "land_surface_water__boundary_type": "1",
        "land_surface_water__boundary_value": "1",
        "soil_surface_water__infiltration_volume_flux": "m s-1",
        "soil_water__effective_porosity": "m m-1",
        "soil_water__pressure_head": "m",
        "soil_water__hydraulic_conductivity": "m s-1",
        "soil_water__volume_fraction": "m/m",
        "land_surface_water__losses_volume_flux": "m s-1",
        "land_surface_water__depth": "m",
        "land_surface_water_surface__elevation": "m",
        "land_surface_water_flow__speed": "m s-1",
        "land_surface_water_flow__azimuth_angle_of_velocity": "degree",
        "land_surface_water__x_component_of_volume_flow_rate": "m2 s-1",
        "land_surface_water__y_component_of_volume_flow_rate": "m2 s-1",
        "soil_surface_water__boundary_volume_flux": "m s-1",
        "soil_surface_water__mean_of_infiltration_volume_flux": "m s-1",
        "land_surface_water__mean_of_precipitation_leq-volume_flux": "m s-1",
        "land_surface_water__mean_of_inflow_volume_flux": "m s-1",
        "land_surface_water__mean_of_losses_volume_flux": "m s-1",
        "land_surface_water__mean_of_drainage_volume_flux": "m s-1",
        "land_surface_water__time_integral_of_error_depth": "m",
        "land_surface_water_flow__froude_number": "1",
    }

    _var_loc = {
        "land_surface__elevation": "face",
        "land_surface_water_flow__manning_n_parameter": "face",
        "land_surface_water__precipitation_leq-volume_flux": "face",
        "land_surface_water__inflow_volume_flux": "face",
        "land_surface_water__boundary_type": "face",
        "land_surface_water__boundary_value": "face",
        "soil_surface_water__infiltration_volume_flux": "face",
        "soil_water__effective_porosity": "face",
        "soil_water__pressure_head": "face",
        "soil_water__hydraulic_conductivity": "face",
        "soil_water__volume_fraction": "face",
        "land_surface_water__losses_volume_flux": "face",
        "land_surface_water__depth": "face",
        "land_surface_water_surface__elevation": "face",
        "land_surface_water_flow__speed": "face",
        "land_surface_water_flow__azimuth_angle_of_velocity": "face",
        "land_surface_water__x_component_of_volume_flow_rate": "edge",
        "land_surface_water__y_component_of_volume_flow_rate": "edge",
        "soil_surface_water__boundary_volume_flux": "face",
        "soil_surface_water__mean_of_infiltration_volume_flux": "face",
        "land_surface_water__mean_of_precipitation_leq-volume_flux": "face",
        "land_surface_water__mean_of_inflow_volume_flux": "face",
        "land_surface_water__mean_of_losses_volume_flux": "face",
        "land_surface_water__mean_of_drainage_volume_flux": "face",
        "land_surface_water__time_integral_of_error_depth": "face",
        "land_surface_water_flow__froude_number": "face",
    }

    def __init__(self):
        """Create a BmiItzi model that is ready for initialization."""
        self.itzi = None

    # Model control functions #

    def initialize(self, filename=None):
        """Initialize the Itzï model.

        Parameters
        ----------
        filename : str, optional
            Path to name of input file.
        """
        self.itzi = itzi.SimulationRunner()
        self.itzi.initialize(conf_file=filename)

    def update(self):
        """Advance model by one time step."""
        self.itzi.step()

    def update_until(self, then):
        """Update model until a particular time.

        Parameters
        ----------
        then : float
            Time to run model until.
        """
        then = timedelta(seconds=float(then))
        self.itzi.sim.update_until(then)

    def finalize(self):
        """Finalize model."""
        self.itzi.finalize()

    # Model information functions #

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    def get_input_var_names(self):
        """Get names of input variables."""
        return list(self._input_var_names.keys())

    def get_output_var_names(self):
        """Get names of output variables."""
        return list(self._output_var_names.keys())

    # Time functions #

    def get_start_time(self):
        """Start time of model."""
        return float(0)

    def get_end_time(self):
        """End time of model."""
        duration = self.itzi.sim.start_time - self.itzi.sim.end_time
        return float(duration.total_seconds())

    def get_current_time(self):
        current_time = self.itzi.sim.sim_time - self.itzi.sim.start_time
        return float(current_time.total_seconds())

    def get_time_step(self):
        return float(self.itzi.sim.dt.total_seconds())

    def get_time_units(self):
        return "s"

    # Variable information functions #

    def get_var_type(self, var_name):
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ptr(var_name).dtype)

    def get_var_units(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Variable units.
        """
        return self._var_units[var_name]

    def get_var_nbytes(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    def get_var_itemsize(self, name):
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name):
        return self._var_loc[name]

    def get_var_grid(self, var_name):
        """Grid id for a variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Grid id.
        """
        for grid_id, grid_name in enumerate(self._var_names.keys()):
            if var_name == grid_name:
                return grid_id

    # Values getting and setting functions #
    def get_value_ptr(self, var_name):
        """Reference to values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Value array.
        """
        internal_var_name = self._var_names[var_name]
        arr = self.itzi.sim.get_array(internal_var_name)
        return arr

    def get_value(self, var_name):
        """Copy of values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Copy of values.
        """
        return self.get_value_ptr(var_name).copy()

    def get_value_at_indices(self, var_name, indices):
        """Get values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        indices : array_like
            Array of indices.

        Returns
        -------
        array_like
            Values at indices.
        """
        return self.get_value_ptr(var_name).take(indices)

    def set_value(self, var_name, src):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        """
        internal_var_name = self._var_names[var_name]
        self.itzi.sim.set_array(internal_var_name, src)

    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        # val = self.get_value_ptr(name)
        # val.flat[inds] = src
        raise NotImplementedError("set_value_at_indices")

    # Grid information functions #
    def get_grid_rank(self, grid_id):
        """Rank of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Rank of grid.
        """
        return len(self.get_grid_shape(grid_id))

    def get_grid_size(self, grid_id):
        """Size of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Size of grid.
        """
        return np.prod(self.get_grid_shape(grid_id))

    def get_grid_shape(self, grid_id):
        """Number of rows and columns of uniform rectilinear grid."""
        for g_id, var_name in enumerate(self._var_names.keys()):
            if g_id == grid_id:
                return self.get_value_ptr(var_name).shape

    def get_grid_spacing(self, grid_id):
        """Spacing of rows and columns of uniform rectilinear grid."""
        return np.array(self.itzi.sim.spacing)

    def get_grid_origin(self, grid_id):
        """Origin of uniform rectilinear grid."""
        return np.array(self.itzi.origin)

    def get_grid_type(self, grid_id):
        """Type of grid."""
        return "uniform_rectilinear"

    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_x(self, grid, x):
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid, y):
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid, z):
        raise NotImplementedError("get_grid_z")
