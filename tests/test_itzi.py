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
    # map_list = gscript.list_grouped('raster', pattern='*out_5by5*')[current_mapset]
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
    wse = gscript.read_command('v.what.rast', map='axis_points', raster='out_mcdo_norain_wse_0004', flags='p')
    df_wse = pd.read_csv(StringIO(wse), sep='|', names=['wse_model'], usecols=[1])
    df_results = mcdo_norain_reference.join(df_wse)
    df_results['abs_error'] = np.abs(df_results['wse_model'] - df_results['wse'])
    mae = np.mean(df_results['abs_error'])
    assert mae < 0.03


def test_mcdo_rain(grass_mcdo_rain_sim, mcdo_rain_reference):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'mcdo_rain'
    wse = gscript.read_command('v.what.rast', map='axis_points', raster='out_mcdo_rain_wse_0001', flags='p')
    df_wse = pd.read_csv(StringIO(wse), sep='|', names=['wse_model'], usecols=[1])
    df_results = mcdo_rain_reference.join(df_wse)
    df_results['abs_error'] = np.abs(df_results['wse_model'] - df_results['wse'])
    mae = np.mean(df_results['abs_error'])
    assert mae < 0.04


def test_ea(ea_test8a_reference, ea_test8a_sim):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'ea8a'
    # Extract results at output points
    itzi_results = gscript.read_command('t.rast.what', points='output_points', strds='out_h',
                                        null_value='*', separator='comma', layout='col')
    # Read results as Pandas dataframe
    col_names = ['Time (min)'] + list(range(1,10))
    df_itzi = pd.read_csv(StringIO(itzi_results), index_col=0, header=1,
                                   names=col_names, na_values='*')
    df_itzi.fillna(0., inplace=True)
    df_itzi.index = pd.to_numeric(df_itzi.index)
    df_itzi.index /= 60.
    del df_itzi['Time (min)']
    abs_error = np.abs(df_itzi - ea_test8a_reference)
    points_values = []
    for pt_idx in range(1,9):
        col_idx = [ea_test8a_reference[pt_idx],
                   df_itzi[pt_idx],
                   abs_error[pt_idx]]
        col_keys = ['lisflood', 'itzi', 'absolute error']
        new_df = pd.concat(col_idx,  axis=1, keys=col_keys)
        new_df.index.name = 'Time (min)'
        # Keep only non null values
        new_df = new_df[new_df.itzi.notnull()]
        points_values.append(new_df)
    for df_err in points_values:
        mae = np.mean(df_err['absolute error'])
        assert mae <= 0.04
