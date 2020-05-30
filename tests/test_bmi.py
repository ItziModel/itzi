#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
import os

import pytest
import pandas as pd
import numpy as np

from itzi import BmiItzi


class TestBmi:

    def test_initialize(self, bmi_object):
        assert hasattr(bmi_object, 'itzi')
        assert hasattr(bmi_object.itzi, 'initialize')

    def test_time_unit(self, bmi_object):
        assert bmi_object.get_time_units() == 's'

    def test_get_component_name(self, bmi_object):
        name = bmi_object.get_component_name()
        assert name == "Itzï"

    def test_get_input_item_count(self, bmi_object):
        count = bmi_object.get_input_item_count()
        assert count == 12

    def test_get_output_item_count(self, bmi_object):
        count = bmi_object.get_output_item_count()
        assert count == 14

    def test_get_input_var_names(self, bmi_object):
        names = bmi_object.get_input_var_names()
        assert len(names) == bmi_object.get_input_item_count()

    def test_get_output_var_names(self, bmi_object):
        names = bmi_object.get_output_var_names()
        assert len(names) == bmi_object.get_output_item_count()


    def test_time_step(self, bmi_object):
        dt = bmi_object.get_time_step()
        assert dt > 0

    def test_start_time(self, bmi_object):
        start_time = bmi_object.get_start_time()
        assert start_time == 0

    def test_current_time(self, bmi_object):
        current_time = bmi_object.get_current_time()
        assert current_time == 0

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
