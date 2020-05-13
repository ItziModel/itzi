#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
from io import StringIO

import pytest
import pandas as pd
import numpy as np
import grass.script as gscript

from itzi import SimulationRunner


def test_number_of_output(grass_5by5_sim):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == '5by5'

    map_list = gscript.list_grouped('raster', pattern='*out_5by5*')[current_mapset]
    print(map_list)
    # fr_map_list = gscript.list_grouped('raster', pattern='*out_5by5_fr_*')[current_mapset]
    # assert len(fr_map_list) == 3
    h_map_list = gscript.list_grouped('raster', pattern='*out_5by5_h_*')[current_mapset]
    assert len(h_map_list) == 4
    v_map_list = gscript.list_grouped('raster', pattern='*out_5by5_v_*')[current_mapset]
    assert len(v_map_list) == 4


def test_mcdo_norain(grass_mcdo_norain_sim, mcdo_norain_reference):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'mcdo_norain'

    map_list = gscript.list_grouped('raster', pattern='out_mcdo_norain_wse*')[current_mapset]
    # print(map_list)
    wse = gscript.read_command('v.what.rast', map='axis_points', raster='out_mcdo_norain_wse_0004', flags='p')
    df_wse = pd.read_csv(StringIO(wse), sep='|', names=['wse_model'], usecols=[1])
    df_results = mcdo_norain_reference.join(df_wse)
    df_results['abs_error'] = np.abs(df_results['wse_model'] - df_results['wse'])
    print(df_results)
    mae = np.mean(df_results['abs_error'])
    assert mae < 0.03


def test_mcdo_rain(grass_mcdo_rain_sim, mcdo_rain_reference):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'mcdo_rain'

    map_list = gscript.list_grouped('raster', pattern='out_mcdo_rain_wse*')[current_mapset]
    print(map_list)
    wse = gscript.read_command('v.what.rast', map='axis_points', raster='out_mcdo_rain_wse_0001', flags='p')
    # print(wse)
    df_wse = pd.read_csv(StringIO(wse), sep='|', names=['wse_model'], usecols=[1])
    # print(mcdo_norain_reference)
    df_results = mcdo_rain_reference.join(df_wse)
    df_results['abs_error'] = np.abs(df_results['wse_model'] - df_results['wse'])
    print(df_results)
    mae = np.mean(df_results['abs_error'])
    assert mae < 0.04
