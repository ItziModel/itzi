#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
import os
from io import StringIO

import pytest
import pandas as pd
import numpy as np
import grass.script as gscript

from itzi import SimulationRunner


def get_rmse(model, ref):
    """return root mean square error"""
    return np.sqrt(np.mean((model - ref)**2))


def get_nse(model, ref):
    """Nash-Sutcliffe Efficiency
    """
    noise = np.mean((ref - model)**2)
    information = np.mean((ref - np.mean(ref))**2)
    return 1-(noise / information)


def get_rsr(model, ref):
    """RMSE/StdDev ratio
    """
    rmse = get_rmse(model, ref)
    return rmse / np.std(ref)


def test_number_of_output(grass_5by5_sim):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == '5by5'
    # map_list = gscript.list_grouped('raster', pattern='*out_5by5*')[current_mapset]
    h_map_list = gscript.list_grouped('raster', pattern='*out_5by5_h_*')[current_mapset]
    assert len(h_map_list) == 4
    wse_map_list = gscript.list_grouped('raster', pattern='*out_5by5_wse_*')[current_mapset]
    assert len(wse_map_list) == 3
    fr_map_list = gscript.list_grouped('raster', pattern='*out_5by5_fr_*')[current_mapset]
    assert len(fr_map_list) == 3
    v_map_list = gscript.list_grouped('raster', pattern='*out_5by5_v_*')[current_mapset]
    assert len(v_map_list) == 4
    vdir_map_list = gscript.list_grouped('raster', pattern='*out_5by5_vdir_*')[current_mapset]
    assert len(vdir_map_list) == 3
    qx_map_list = gscript.list_grouped('raster', pattern='*out_5by5_qx_*')[current_mapset]
    assert len(qx_map_list) == 3
    qy_map_list = gscript.list_grouped('raster', pattern='*out_5by5_qy_*')[current_mapset]
    assert len(qy_map_list) == 3
    verr_map_list = gscript.list_grouped('raster', pattern='*out_5by5_verror_*')[current_mapset]
    assert len(verr_map_list) == 3



def test_flow_symmetry(grass_5by5_sim):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == '5by5'
    h_values = gscript.read_command('v.what.rast', map='control_points', flags='p', raster='out_5by5_h_0001')
    s_h = pd.read_csv(StringIO(h_values), sep='|', names=['h'], usecols=[1], squeeze=True)
    print(s_h)
    print(np.isclose(s_h[:-1], s_h[1:]))
    assert np.all(np.isclose(s_h[:-1], s_h[1:]))


def test_region_mask(grass_5by5, test_data_path):
    """Check if temporary mask and region are set and teared down.
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == '5by5'
    # Get data from initial region and mask
    init_ncells = int(gscript.parse_command('g.region', flags='pg')['cells'])
    init_nulls = int(gscript.parse_command('r.univar', map='z', flags='g')['null_cells'])
    # Set simulation (should set region and mask)
    config_file = os.path.join(test_data_path, '5by5', '5by5_mask.ini')
    sim_runner = SimulationRunner()
    sim_runner.initialize(config_file)
    # Run simulation
    sim_runner.run().finalize()
    # Check temporary mask and region
    assert int(gscript.parse_command('r.univar', map='out_5by5_v_max', flags='g')['n']) == 9
    # Check tear down
    assert int(gscript.parse_command('g.region', flags='pg')['cells']) == init_ncells
    assert int(gscript.parse_command('r.univar', map='z', flags='g')['null_cells']) == init_nulls
    return sim_runner


def test_mcdo_norain(grass_mcdo_norain_sim, mcdo_norain_reference):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'mcdo_norain'

    map_list = gscript.list_grouped('raster', pattern='out_mcdo_norain_wse*')[current_mapset]
    wse = gscript.read_command('v.what.rast', map='axis_points',
                               raster='out_mcdo_norain_wse_0004', flags='p')
    df_wse = pd.read_csv(StringIO(wse), sep='|', names=['wse_model'], usecols=[1])
    df_results = mcdo_norain_reference.join(df_wse)
    df_results['abs_error'] = np.abs(df_results['wse_model'] - df_results['wse'])
    mae = np.mean(df_results['abs_error'])
    assert mae < 0.03


def test_flow_is_unidimensional(grass_mcdo_norain_sim):
    """In the MacDonald 1D test, flow should be unidimensional in x dimension
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'mcdo_norain'

    map_list = gscript.list_grouped('raster', pattern='out_mcdo_norain_qy*')[current_mapset]
    for raster in map_list:
        univar = gscript.parse_command('r.univar', map=raster, flags='g')
        assert float(univar['min']) == 0
        assert float(univar['max']) == 0


def test_mcdo_rain(grass_mcdo_rain_sim, mcdo_rain_reference):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'mcdo_rain'
    wse = gscript.read_command('v.what.rast', map='axis_points', raster='out_mcdo_rain_wse_0001', flags='p')
    df_wse = pd.read_csv(StringIO(wse), sep='|', names=['wse_model'], usecols=[1])
    df_results = mcdo_rain_reference.join(df_wse)
    df_results['abs_error'] = np.abs(df_results['wse_model'] - df_results['wse'])
    mae = np.mean(df_results['abs_error'])
    assert mae < 0.04


def test_ea8a(ea_test8a_reference, ea_test8a_sim):
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
    # Compute the absolute error
    abs_error = np.abs(df_itzi - ea_test8a_reference)
    # Compute MAE for each point
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
    # Check if MAE is below threshold
    for df_err in points_values:
        mae = np.mean(df_err['absolute error'])
        assert mae <= 0.04


def test_ea8b(ea_test8b_reference, ea_test8b_sim):
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'ea8b'
    # Extract results at output points
    itzi_results = gscript.read_command('t.rast.what', points='manhole_location',
                                        strds='out_drainage_stats', null_value='*',
                                        separator='comma', layout='col')
    # Read results as Pandas series
    series_itzi = pd.read_csv(StringIO(itzi_results), index_col=0, header=None,
                              squeeze=True, names=['time', 'itzi'])
    # convert to timedelta
    series_itzi.index = pd.to_timedelta(series_itzi.index, unit='s')
    # force temporal resolution
    series_itzi = series_itzi.resample(rule='s').interpolate(method='time').resample('30s').mean()
    series_itzi.name = 'itzi'
    # Convert from m/s to m3/s (2m resolution)
    series_itzi = series_itzi * 2*2
    # Create a DataFrame with two columns and drop NaN
    df_values = pd.concat([series_itzi, ea_test8b_reference], axis=1).dropna()
    df_values.to_csv('/home/laurent/eat8b.results.csv')
    # Compare results
    rsr = get_rsr(df_values.itzi, df_values.reference)
    nse = get_nse(df_values.itzi, df_values.reference)
    # print(f'nse {nse}')
    # print(f'rsr {rsr}')
    assert rsr < 0.1
    assert nse > 0.99
