#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" """

import os

import pytest
import numpy as np

from itzi import BmiItzi


@pytest.fixture(scope="class")
def bmi_object(grass_5by5, test_data_path):
    itzi_bmi = BmiItzi()
    conf_file = os.path.join(test_data_path, "5by5", "5by5.ini")
    itzi_bmi.initialize(conf_file)
    return itzi_bmi


class TestBmi:
    # Model control functions #
    def test_initialize(self, bmi_object):
        assert hasattr(bmi_object, "itzi")
        assert hasattr(bmi_object.itzi, "initialize")

    def test_update(self, bmi_object):
        old_time = bmi_object.get_current_time()
        bmi_object.update()
        dt = bmi_object.get_time_step()
        new_time = bmi_object.get_current_time()
        assert new_time == old_time + dt

    def test_update_until(self, bmi_object):
        old_time = bmi_object.get_current_time()
        then = 2
        bmi_object.update_until(then)
        new_time = bmi_object.get_current_time()
        assert new_time == old_time + then

    def test_finalize(self, bmi_object):
        bmi_object.finalize()

    # Model information functions #
    def test_get_component_name(self, bmi_object):
        name = bmi_object.get_component_name()
        assert name == "Itzï"

    def test_get_input_item_count(self, bmi_object):
        count = bmi_object.get_input_item_count()
        assert count == 13

    def test_get_output_item_count(self, bmi_object):
        count = bmi_object.get_output_item_count()
        assert count == 14

    def test_get_input_var_names(self, bmi_object):
        names = bmi_object.get_input_var_names()
        assert len(names) == bmi_object.get_input_item_count()

    def test_get_output_var_names(self, bmi_object):
        names = bmi_object.get_output_var_names()
        assert len(names) == bmi_object.get_output_item_count()

    # Time functions #
    def test_time_unit(self, bmi_object):
        assert bmi_object.get_time_units() == "s"

    def test_time_step(self, bmi_object):
        dt = bmi_object.get_time_step()
        assert dt > 0

    def test_start_time(self, bmi_object):
        start_time = bmi_object.get_start_time()
        assert start_time == 0

    def test_current_time(self, bmi_object):
        current_time = bmi_object.get_current_time()
        assert current_time == 0

    # Variable information functions #
    def test_get_var_type(self, bmi_object):
        var_type = bmi_object.get_var_type("land_surface__elevation")
        assert var_type == "float32"

    def test_get_var_units(self, bmi_object):
        var_unit = bmi_object.get_var_units("land_surface__elevation")
        assert var_unit == "m"

    def test_get_var_nbytes(self, bmi_object):
        var_name = "land_surface__elevation"
        var_nbytes = bmi_object.get_var_nbytes(var_name)
        grid_id = bmi_object.get_var_grid(var_name)
        grid_size = bmi_object.get_grid_size(grid_id)
        item_size = bmi_object.get_var_itemsize(var_name)
        assert var_nbytes == grid_size * item_size

    def test_get_var_itemsize(self, bmi_object):
        itemsize = bmi_object.get_var_itemsize("land_surface__elevation")
        assert itemsize == 32 / 8

    def test_get_var_location(self, bmi_object):
        loc = bmi_object.get_var_location("land_surface__elevation")
        assert loc == "face"

    def test_get_var_grid(self, bmi_object):
        grid_id = bmi_object.get_var_grid("land_surface__elevation")
        assert grid_id == 0

    # Values getting and setting functions #
    def test_get_value_ptr(self, bmi_object):
        value_ptr = bmi_object.get_value_ptr("land_surface__elevation")
        ref_value = bmi_object.itzi.sim.get_array("dem")
        assert np.all(value_ptr == ref_value)

    def test_get_value(self, bmi_object):
        value = bmi_object.get_value("land_surface__elevation")
        ref_value = bmi_object.itzi.sim.get_array("dem")
        assert np.all(value == ref_value)

    def test_get_value_at_indices(self, bmi_object):
        value = bmi_object.get_value_at_indices("land_surface__elevation", 2)
        print(value)
        assert value.item() == 0

    def test_set_value(self, bmi_object):
        var_name = "land_surface__elevation"
        grid_id = bmi_object.get_var_grid(var_name)
        grid_shape = bmi_object.get_grid_shape(grid_id)
        value = np.ones(grid_shape)
        assert np.all(value != bmi_object.get_value_ptr(var_name))
        bmi_object.set_value(var_name, value)
        assert np.all(value == bmi_object.get_value_ptr(var_name))

    # Grid information functions #
    def test_get_grid_rank(self, bmi_object):
        grid_id = bmi_object.get_var_grid("land_surface__elevation")
        grid_rank = bmi_object.get_grid_rank(grid_id)
        assert grid_rank == 2

    def test_get_grid_size(self, bmi_object):
        grid_id = bmi_object.get_var_grid("land_surface__elevation")
        grid_size = bmi_object.get_grid_size(grid_id)
        assert grid_size == 25

    def test_get_grid_shape(self, bmi_object):
        grid_id = bmi_object.get_var_grid("land_surface__elevation")
        grid_shape = bmi_object.get_grid_shape(grid_id)
        assert grid_shape == (5, 5)

    def test_get_grid_spacing(self, bmi_object):
        grid_id = bmi_object.get_var_grid("land_surface__elevation")
        grid_spacing = bmi_object.get_grid_spacing(grid_id)
        assert isinstance(grid_spacing, np.ndarray)
        reference = np.array([10, 10])
        assert np.all(reference == grid_spacing)

    def test_get_grid_origin(self, bmi_object):
        grid_id = bmi_object.get_var_grid("land_surface__elevation")
        grid_origin = bmi_object.get_grid_origin(grid_id)
        assert isinstance(grid_origin, np.ndarray)
        reference = np.array([50, 0])
        assert np.all(reference == grid_origin)

    def test_get_grid_type(self, bmi_object):
        grid_id = bmi_object.get_var_grid("land_surface__elevation")
        grid_type = bmi_object.get_grid_type(grid_id)
        assert grid_type == "uniform_rectilinear"
