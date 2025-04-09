"""
Integration tests using the EA test case 8b.
The results from itzi are compared with the those from XPSTORM.
"""

import os
import zipfile
from io import StringIO
from configparser import ConfigParser

import numpy as np
import pandas as pd
import pytest
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


def roughness(timeseries):
    """Sum of the squared difference of
    the normalized differences.
    """
    f = timeseries.diff()
    normed_f = (f - f.mean()) / f.std()
    return (normed_f.diff() ** 2).sum()


@pytest.fixture(scope="class")
def ea_test8b(grass_xy_session, ea_test_files):
    """Create the GRASS env for ea test 8a.
    """
    # Unzip the file
    file_path = os.path.join(ea_test_files, 'Test8B_dataset_2010.zip')
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(ea_test_files)
    unzip_path = os.path.join(ea_test_files, 'Test8B dataset 2010')
    # Create new mapset
    mapset_name = 'ea8b'
    gscript.run_command('g.mapset', mapset=mapset_name, flags='c')
    # Define the region
    region = gscript.parse_command('g.region', res=2,
                                   s=664408, w=263976,
                                   e=264940, n=664808, flags='g')
    assert int(region['rows']) == 200
    assert int(region['cols']) == 482
    # DEM
    dem_path = os.path.join(unzip_path, 'Test8DEM.asc')
    gscript.run_command('r.in.gdal', input=dem_path, output='dem50cm')
    gscript.run_command('r.resamp.stats', input='dem50cm', output='dem2m')
    univar_dem = gscript.parse_command('r.univar', map='dem2m', flags='g')
    assert int(univar_dem['null_cells']) == 0
    # Buildings
    buildings_path = os.path.join(unzip_path, 'Test8Buildings.asc')
    gscript.run_command('r.in.gdal', input=buildings_path, output='buildings')
    gscript.mapcalc('dem2m_buildings=if(isnull(buildings), dem2m, dem2m+5)')
    # Manning
    road_path = os.path.join(unzip_path, 'Test8RoadPavement.asc')
    gscript.run_command('r.in.gdal', input=road_path, output='road50cm')
    gscript.mapcalc('n=if(isnull(road50cm), 0.05, 0.02)')
    univar_n = gscript.parse_command('r.univar', map='n', flags='g')
    assert float(univar_n['min']) == 0.02
    assert float(univar_n['max']) == 0.05
    assert int(univar_n['null_cells']) == 0
    # Output points #
    stages_path = os.path.join(unzip_path, 'Test8Output.csv')
    gscript.run_command('v.in.ascii', input=stages_path, output='output_points',
                        format='point', sep='comma', skip=1, cat=1, x=2, y=3)
    # Manhole location
    gscript.write_command('v.in.ascii', input='-',
                          stdin='264895|664747',
                          output='manhole_location')
    return None


@pytest.fixture(scope="session")
def ea_test8b_reference(test_data_path):
    """Take the results from xpstorm as reference.
    """
    col_names = ['Time', 'results']
    file_path = os.path.join(test_data_path, 'EA_test_8', 'b', 'xpstorm.csv')
    df_ref = pd.read_csv(file_path, index_col=0, names=col_names)
    # Convert to seconds
    df_ref.index *= 60.
    # Round time to 10 ms
    df_ref.index = df_ref.index.round(decimals=2)
    # convert indices to timedelta
    df_ref.index = pd.to_timedelta(df_ref.index, unit='s')
    # to series
    ds_ref= df_ref.squeeze()
    return ds_ref


@pytest.fixture(scope="class")
def ea_test8b_sim(ea_test8b, test_data_path):
    """
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'ea8b'
    inp_file = os.path.join(test_data_path, 'EA_test_8', 'b',
                            'test8b_drainage_ponding.inp')
    config_dict = {'time': {'duration': '03:20:00', 'record_step': '00:00:30'},
                   'input': {'dem': 'dem2m_buildings', 'friction': 'n'},
                   'output': {'prefix': 'out', 'values': 'h, drainage_stats'},
                   'options': {'theta': 0.7, 'cfl': 0.5},
                   'drainage': {'swmm_inp': inp_file, 'orifice_coeff': 1, 'output': 'out_drainage'}}
    parser = ConfigParser()
    parser.read_dict(config_dict)
    conf_file = os.path.join(test_data_path, 'EA_test_8', 'b',
                            'ea2dt8b.ini')
    with open(conf_file, 'w') as f:
        parser.write(f)
    sim_runner = SimulationRunner()
    sim_runner.initialize(conf_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="class")
def ea8b_itzi_drainage_results(ea_test8b_sim):
    """Extract linkage flow from the drainage network
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'ea8b'
    select_col = ['start_time', 'linkage_flow']
    itzi_results = gscript.read_command('t.vect.db.select', input='out_drainage')
    # translate to Pandas dataframe and keep only linkage_flow with start_time over 3000
    df_itzi_results = pd.read_csv(StringIO(itzi_results), sep='|')[select_col]
    df_itzi_results = df_itzi_results[df_itzi_results.start_time >= 3000]
    df_itzi_results.set_index('start_time', drop=True, inplace=True, verify_integrity=True)
    # convert indices to timedelta
    df_itzi_results.index = pd.to_timedelta(df_itzi_results.index, unit='s')
    # to series
    ds_itzi_results = df_itzi_results.squeeze()
    return ds_itzi_results


def test_ea8b(ea_test8b_reference, ea8b_itzi_drainage_results):
    """Compare results with XPSTORM
    """
    # Extract results at output points
    # itzi_results = gscript.read_command('t.rast.what', points='manhole_location',
    #                                     strds='out_drainage_stats', null_value='*',
    #                                     separator='comma', layout='col')
    ds_itzi_results = ea8b_itzi_drainage_results
    ds_ref = ea_test8b_reference
    # calculate NSE
    nse = get_nse(ds_itzi_results, ds_ref)
    rsr = get_rsr(ds_itzi_results, ds_ref)
    itzi_roughness = roughness(ds_itzi_results)
    xpstorm_roughness = roughness(ds_ref)
    autocorrelation_itzi = ds_itzi_results.autocorr(lag=1)
    autocorrelation_ref = ds_ref.autocorr(lag=1)
    print(f"xpstorm roughness (better closer to zero): {xpstorm_roughness}")
    print(f"xpstorm autocorrelation (better closer to one): {autocorrelation_ref}")
    print(f"flow exchange roughness (better closer to zero): {itzi_roughness}")
    print(f"flow exchange autocorrelation (better closer to one): {autocorrelation_itzi}")
    print(f"{nse=}")
    print(f"{rsr=}")
    assert nse > 0.99
    assert rsr < 0.01
    assert itzi_roughness < 0.2
    assert autocorrelation_itzi > 0.9