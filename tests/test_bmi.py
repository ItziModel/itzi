#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
import os

import pytest
import pandas as pd
import numpy as np
import grass.script as gscript

from itzi import BmiItzi

def test_initialize(test_data_path, grass_5by5):
    itzi_bmi = BmiItzi()
    conf_file = os.path.join(test_data_path, '5by5', '5by5.ini')
    itzi_bmi.initialize(conf_file)
    assert hasattr(itzi_bmi, 'itzi')
