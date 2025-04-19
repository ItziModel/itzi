"""Test the drainage component.
"""
import os
from configparser import ConfigParser
from io import StringIO

import pandas as pd
import pytest
import grass.script as gscript

from itzi import SimulationRunner


@pytest.fixture(scope="class")
def drainage_sim_results(grass_xy_session, test_data_path):
    # Create new mapset
    gscript.run_command('g.mapset', mapset='drainage', flags='c')

    array_shape = (9, 9)
    cell_size = 5
    # define region
    gscript.run_command('g.region', res=cell_size,
                        s=0,
                        n=array_shape[1] * cell_size,
                        w=0,
                        e=array_shape[0] * cell_size,
                        flags='o')
    # create maps
    gscript.mapcalc('dem=100')
    gscript.mapcalc('friction=0.03')
    # boundary conditions
    boundary_conditions = os.path.join(test_data_path, 'test_drainage_bc.asc')
    gscript.run_command('v.in.ascii', input=boundary_conditions, output='boundary_conditions')
    gscript.run_command('v.to.rast', input='boundary_conditions', type='point',
                        output='bctype', use='val', value=4)
    gscript.run_command('v.to.rast', input='boundary_conditions', type='point',
                        output='bcvalue', use='val', value=0)
    # SWMM config file based on EA test 8b
    inp_file = os.path.join(test_data_path, 'test_drainage.inp')
    # Create itzi config file
    config_dict = {'time': {'duration': '03:20:00', 'record_step': '00:00:30'},
                   'input': {'dem': 'dem', 'friction': 'friction'},
                   'output': {'prefix': 'out', 'values': 'h, drainage_stats'},
                   'options': {'theta': 0.7, 'cfl': 0.5},
                   'drainage': {'swmm_inp': inp_file, 'orifice_coeff': 1, 'output': 'out_drainage'}}

    conf_file = os.path.join(test_data_path, 'test_drainage.ini')
    parser = ConfigParser()
    parser.read_dict(config_dict)
    with open(conf_file, 'w') as f:
        parser.write(f)
    
    # Create and run simulation
    sim_runner = SimulationRunner()
    sim_runner.initialize(conf_file)
    sim_runner.run().finalize()

    # Retrieve results
    select_col = ['start_time', 'linkage_flow']
    itzi_results = gscript.read_command('t.vect.db.select', input='out_drainage')
    # translate to Pandas dataframe and keep only linkage_flow with start_time over 3000
    df_itzi_results = pd.read_csv(StringIO(itzi_results), sep='|')[select_col]
    df_itzi_results = df_itzi_results[df_itzi_results.start_time >= 3000]
    df_itzi_results.set_index('start_time', drop=True, inplace=True, verify_integrity=True)
    # convert indices to timedelta
    df_itzi_results.index = pd.to_timedelta(df_itzi_results.index, unit='s')
    # to series
    return df_itzi_results.squeeze()


def test_drainage_coupling_stability(drainage_sim_results, helpers):
    """Test the stability of the coupling between itzi and SWMM.
    """
    roughness = helpers.roughness(drainage_sim_results)
    autocorrelation = drainage_sim_results.autocorr(lag=1)
    # assert roughness < 5
    assert autocorrelation > 0.9
