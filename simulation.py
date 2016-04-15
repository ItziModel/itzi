# coding=utf8
"""
Copyright (C) 2015-2016  Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
from __future__ import division
import csv
import warnings
from datetime import datetime, timedelta
import numpy as np
try:
    import bottleneck as bn
except ImportError:
    bn = np

import domain
import gis
import flow
import infiltration
from swmm.swmm import Swmm5, SwmmNode
from itzi_error import NullError


class SuperficialFlowSimulation(object):
    """Manage the general simulation:
    - update input values for each time-step
    - trigger the writing of results and statistics
    Accessed via the run() method
    """

    def __init__(self, record_step, input_maps, output_maps,
                 dtype=np.float32,
                 stats_file=None,
                 start_time=datetime(1, 1, 1),
                 end_time=datetime(1, 1, 1),
                 sim_duration=timedelta(0),
                 sim_param=None,
                 swmm_params=None):
        assert isinstance(start_time, datetime), \
            "start_time not a datetime object!"
        assert isinstance(sim_duration, timedelta), \
            "sim_duration not a timedelta object!"
        assert sim_duration >= timedelta(0), "sim_duration is negative!"
        assert isinstance(record_step, timedelta), \
            "record_step not a timedelta object!"
        assert record_step > timedelta(0), "record_step must be > 0"
        assert sim_param is not None

        self.record_step = record_step
        self.start_time = start_time
        self.set_duration(end_time, sim_duration)
        # set simulation time to start_time a the beginning
        self.sim_time = self.start_time
        self.dt = sim_param['dtmax']
        self.dtinf = sim_param['dtinf']
        # set temporal type of results
        self.set_temporal_type()
        self.in_map_names = input_maps
        self.out_map_names = output_maps
        self.swmm_params=swmm_params

        self.dtype = dtype
        # instantiate a Igis object
        self.gis = gis.Igis(start_time=self.start_time,
                            end_time=self.end_time,
                            dtype=dtype, mkeys=self.in_map_names.keys())
        self.gis.msgr.verbose(_(u"Reading GIS..."))
        self.gis.read(self.in_map_names)

        # Determine infiltration type by checking if const. infiltr. is given
        # Coherence of given input maps is checked upstream
        if self.in_map_names['in_inf']:
            self.inftype = 'fix'
            self.infiltration = infiltration.InfConstantRate()
        elif self.in_map_names['in_cap_pressure']:
            self.inftype = 'ga'
            self.infiltration = infiltration.InfGreenAmpt(self.gis.xr,
                                                          self.gis.yr)
        else:
            self.inftype = None
            self.infiltration = infiltration.InfNull()
        # instantiate an array with defult at zero
        self.arr_inf = self.zeros_array()

        # dict to store TimedArrays objects
        self.tarrays = dict.fromkeys(self.in_map_names.keys())
        # Populate it
        self.create_timed_arrays()
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        # Instantiate Massbal object
        self.massbal = None
        if stats_file:
            dom_size = self.gis.yr * self.gis.xr
            self.massbal = MassBal(dom_size=dom_size, file_name=stats_file)
        # mask array
        self.mask = np.full(shape=(self.gis.yr, self.gis.xr),
                            fill_value=False, dtype=np.bool_)

        # simulation parameters
        self.sim_param = sim_param

        # SWMM5 integration
        self.as_drainage = False
        if all(self.swmm_params.itervalues()):
            self.as_drainage = True
            self.set_drainage_model()

    def set_duration(self, end_time, sim_duration):
        """If sim_duration is given, end_time is ignored
        This is normally checked upstream
        """
        if not sim_duration:
            self.duration = end_time - self.start_time
            self.end_time = end_time
        else:
            self.duration = sim_duration
            self.end_time = self.start_time + sim_duration
        return self

    def set_temporal_type(self):
        """A start_time equal to datetime.min means user did not
        provide a start_time. Therefore a relative temporal type
        is set for results writing.
        It's a potential problem in case a simulation is set to actually
        starts on 0001-01-01 00:00, as the results will be written
        in relative STDS.
        """
        self.temporal_type = 'absolute'
        if self.start_time == datetime.min:
            self.temporal_type = 'relative'
        return self

    def create_timed_arrays(self):
        """Create TimedArray objects and store them in the tarrays dict
        """
        self.tarrays['in_z'] = TimedArray('in_z', self.gis, self.zeros_array)
        self.tarrays['in_n'] = TimedArray('in_n', self.gis, self.zeros_array)
        self.tarrays['in_h'] = TimedArray('in_h', self.gis, self.zeros_array)
        self.tarrays['in_inf'] = TimedArray('in_inf',
                                            self.gis, self.zeros_array)
        self.tarrays['in_eff_por'] = TimedArray('in_eff_por',
                                                self.gis, self.zeros_array)
        self.tarrays['in_cap_pressure'] = TimedArray('in_cap_pressure',
                                                     self.gis,
                                                     self.zeros_array)
        self.tarrays['in_hyd_conduct'] = TimedArray('in_hyd_conduct',
                                                    self.gis,
                                                    self.zeros_array)
        self.tarrays['in_rain'] = TimedArray('in_rain',
                                             self.gis, self.zeros_array)
        self.tarrays['in_q'] = TimedArray('in_q',
                                          self.gis, self.zeros_array)
        self.tarrays['in_bcval'] = TimedArray('in_bcval',
                                              self.gis, self.zeros_array)
        # Default of bctypes is an int array filled with ones
        self.tarrays['in_bctype'] = TimedArray('in_bctype',
                                               self.gis, self.ones_array)
        return self

    def set_drainage_model(self):
        """create python swmm object
        open the project files
        get list of nodes
        create a list of linkable nodes
        """
        self.swmm = Swmm5(swmm_so='./source/swmm5.so')
        self.swmm.open(input_file = self.swmm_params['input'],
                       report_file = self.swmm_params['report'],
                       output_file = self.swmm_params['output'])

    def run(self):
        """Perform a full superficial flow simulation
        including recording of data and mass_balance calculation
        """
        # Instantiate SurfaceDomain object
        start_h_masked = self.mask_array(
            self.tarrays['in_h'].get_array(self.start_time), 0)
        assert not np.any(np.isnan(start_h_masked))
        rast_dom = domain.SurfaceDomain(
                dx=self.gis.dx,
                dy=self.gis.dy,
                arr_h=start_h_masked,
                arr_def=self.zeros_array(),
                hf_min=self.sim_param['hmin'],
                theta=self.sim_param['theta'],
                dtmax=self.sim_param['dtmax'],
                a=self.sim_param['cfl'],
                slope_threshold=self.sim_param['slmax'],
                v_routing=self.sim_param['vrouting'])
        record_counter = 1
        last_inf = 0.
        duration_s = self.duration.total_seconds()

        self.gis.msgr.verbose(_(u"Starting time-stepping..."))
        while self.sim_time < self.end_time:
            # display advance of simulation
            self.gis.msgr.percent(rast_dom.sim_clock, duration_s, 1)

            # Calculate when will happen the next records writing
            next_record = record_counter * self.record_step.total_seconds()

            # calculate infiltration
            self.set_inf_forced_timestep(next_record)
            self.infiltration.set_dt(self.dtinf, rast_dom.sim_clock,
                                     self.inf_forced_ts)
            if last_inf + self.infiltration.dt >= rast_dom.sim_clock:
                self.calculate_infiltration(rast_dom.arr_h)
                last_inf = float(rast_dom.sim_clock)

            # update arrays
            self.update_domain_arrays(rast_dom)
            # next forced flow time-step
            next_ts = self.next_forced_timestep()
            # step() raise NullError in case of NaN/NULL cell
            # if this happen, stop simulation and
            # output a map showing the errors
            try:
                rast_dom.step(next_ts, self.massbal)
            except NullError:
                self.write_error_to_gis(rast_dom.arr_h, rast_dom.arr_err)
                self.gis.msgr.fatal(_(u"Null value detected in simulation at time {}, terminating").format(self.sim_time))
            # update simulation time and dt
            self.sim_time = (self.start_time +
                             timedelta(seconds=rast_dom.sim_clock))
            self.dt = rast_dom.dt
            if self.massbal:
                self.massbal.add_value('tstep', self.dt)
            # write simulation results
            rec_time = rast_dom.sim_clock / self.record_step.total_seconds()
            if rec_time >= record_counter:
                self.gis.msgr.verbose(_(u"Writting output map..."))
                self.output_arrays = rast_dom.get_output_arrays(self.out_map_names)
                self.write_results_to_gis(record_counter)
                record_counter += 1
                if self.massbal:
                    self.write_mass_balance(rast_dom.sim_clock)
        # register generated maps in GIS
        self.register_results_in_gis()
        if self.out_map_names['out_h']:
            self.write_hmax_to_gis(rast_dom.arr_hmax)
        return self

    def write_mass_balance(self, sim_clock):
        '''
        '''
        if self.temporal_type == 'absolute':
            self.massbal.write_values(self.sim_time)
        elif self.temporal_type == 'relative':
            self.massbal.write_values(sim_clock)
        else:
            assert False, "unknown temporal type!"
        return self

    def zeros_array(self):
        """return a np array of the domain dimension, filled with zeros
        dtype is set to object's dtype.
        Intended to be used as default for most of the input model maps
        """
        return np.zeros(shape=(self.gis.yr, self.gis.xr), dtype=self.dtype)

    def ones_array(self):
        """return a np array of the domain dimension, filled with ones
        dtype is set to unsigned integer.
        Intended to be used as default for bctype map
        """
        return np.ones(shape=(self.gis.yr, self.gis.xr), dtype=self.dtype)

    def next_forced_timestep(self):
        """return the future time in seconds at which the superficial flow
        model will be forced to step.
        default to next forced time-step of infiltration model
        """
        return self.inf_forced_ts

    def set_inf_forced_timestep(self, next_record):
        """Given a next superficial flow time-step in seconds as entry,
        return the future time in seconds at which the infiltration
        model will be forced to step.
        """
        self.inf_forced_ts = min(next_record, self.duration.total_seconds())
        return self

    def update_mask(self, arr_z):
        '''Create a mask array marking NULL values.
        '''
        self.mask[:] = np.isnan(arr_z)
        return self

    def mask_array(self, arr, default_value):
        '''Replace NULL values in the input array by the default_value
        '''
        mask = np.logical_or(np.isnan(arr), self.mask)
        arr[mask] = default_value
        return arr

    def unmask_array(self, arr):
        '''Replace values in the input array by NULL values from mask
        '''
        unmasked_array = np.copy(arr)
        unmasked_array[self.mask] = np.nan
        return unmasked_array

    def calculate_infiltration(self, arr_h):
        """Calculate an array of infiltration rates in mm/h
        """
        if self.inftype == 'fix':
            self.infiltration.update_input(
                self.tarrays['in_inf'].get_array(self.sim_time))
            self.arr_inf[:] = self.infiltration.get_inf_rate(arr_h)
        elif self.inftype == 'ga':
            self.infiltration.update_input(
                self.tarrays['in_eff_por'].get_array(self.sim_time),
                self.tarrays['in_cap_pressure'].get_array(self.sim_time),
                self.tarrays['in_hyd_conduct'].get_array(self.sim_time))
            self.arr_inf[:] = self.infiltration.get_inf_rate(arr_h)
        elif self.inftype is None:
            pass  # arr_inf is set to zero at init
        else:
            assert False, "unknown infiltration type"


    def update_domain_arrays(self, rast_dom):
        """Takes a SurfaceDomain object as input
        get new array of the given object using TimedArray
        Replace the NULL values
        set new domain arrays
        """
        assert isinstance(rast_dom, domain.SurfaceDomain), \
            "rast_dom not the expected object!"

        sim_time = self.start_time + timedelta(seconds=rast_dom.sim_clock)
        # DEM
        if not self.tarrays['in_z'].is_valid(sim_time):
            arr_z = self.tarrays['in_z'].get_array(sim_time)
            self.update_mask(arr_z)
            arr_z[:] = self.mask_array(arr_z, np.finfo(self.dtype).max)
            assert not np.any(np.isnan(arr_z))
            rast_dom.arr_z = arr_z
            rast_dom.update_flow_dir()
        # Friction
        if not self.tarrays['in_n'].is_valid(sim_time):
            arr_n = self.tarrays['in_n'].get_array(sim_time)
            arr_n[:] = self.mask_array(arr_n, 1)
            assert not np.any(np.isnan(arr_n))
            rast_dom.arr_n = arr_n
        # Boundary conditions values
        if not self.tarrays['in_bcval'].is_valid(sim_time):
            arr_bcval = self.tarrays['in_bcval'].get_array(sim_time)
            arr_bcval[:] = self.mask_array(arr_bcval, 0)
            assert not np.any(np.isnan(arr_bcval))
            rast_dom.arr_bcval = arr_bcval
        # Boundary conditions types. Replace NULL by 1 (closed boundary)
        if not self.tarrays['in_bctype'].is_valid(sim_time):
            arr_bctype = self.tarrays['in_bctype'].get_array(sim_time)
            arr_bctype[:] = self.mask_array(arr_bctype, 1)
            assert not np.any(np.isnan(arr_bctype))
            rast_dom.arr_bctype = arr_bctype
        # External values array
        arr_ext = self.set_ext_array(
            in_q=self.tarrays['in_q'].get_array(sim_time),
            in_rain=self.tarrays['in_rain'].get_array(sim_time),
            in_inf=self.arr_inf)
        arr_ext[:] = self.mask_array(arr_ext, 0)
        assert not np.any(np.isnan(arr_ext))
        rast_dom.arr_ext = arr_ext
        return self

    def set_ext_array(self, in_q, in_rain, in_inf):
        """Combine rain, infiltration etc. into a unique array in m/s
        rainfall and infiltration are considered in mm/h
        send relevant values to MassBal
        """
        assert isinstance(in_q, np.ndarray), "not a np array!"
        assert isinstance(in_rain, np.ndarray), "not a np array!"
        assert isinstance(in_inf, np.ndarray), "not a np array!"

        mmh_to_ms = 1000. * 3600.
        # mass balance in m3
        if self.massbal:
            surf_dt = self.gis.dx * self.gis.dy * self.dt
            rain_vol = (bn.nansum(in_rain[np.logical_not(self.mask)]) /
                        mmh_to_ms * surf_dt)
            inf_vol = (bn.nansum(in_inf[np.logical_not(self.mask)]) /
                       mmh_to_ms * surf_dt)
            inflow_vol = bn.nansum(in_q[np.logical_not(self.mask)]) * surf_dt
            self.massbal.add_value('rain_vol', rain_vol)
            self.massbal.add_value('inf_vol', inf_vol)
            self.massbal.add_value('inflow_vol', inflow_vol)

        arr_ext = np.copy(in_q)
        flow.set_ext_array(in_q, in_rain, in_inf, arr_ext, mmh_to_ms)
        return arr_ext

    def write_results_to_gis(self, record_counter):
        """Format the name of each maps using the record number as suffix
        Send a couple array, name to the GIS writing function.
        """
        for k, arr in self.output_arrays.iteritems():
            if isinstance(arr, np.ndarray):
                suffix = str(record_counter).zfill(6)
                map_name = "{}_{}".format(self.out_map_names[k], suffix)
                arr_unmasked = self.unmask_array(arr)
                # Export depth if above hfmin. If not, export NaN
                if k == 'out_h':
                    hfmin = self.sim_param['hmin']
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arr_unmasked[arr_unmasked <= hfmin] = np.nan
                # write the raster
                self.gis.write_raster_map(arr_unmasked, map_name, k)
                # add map name and time to the corresponding list
                self.output_maplist[k].append((map_name, self.sim_time))
        return self

    def write_error_to_gis(self, arr_h, arr_error):
        '''Write a given depth array and boolean error array to the GIS
        '''
        map_h_name = "{}_error".format(self.out_map_names['out_h'])
        self.gis.write_raster_map(arr_h, map_h_name, 'out_h')
        # add map name to the revelant list
        self.output_maplist['out_h'].append(map_h_name)
        return self

    def write_hmax_to_gis(self, arr_hmax):
        '''Write a given depth array to the GIS
        '''
        arr_hmax_unmasked = self.unmask_array(arr_hmax)
        map_hmax_name = "{}_max".format(self.out_map_names['out_h'])
        self.gis.write_raster_map(arr_hmax_unmasked, map_hmax_name, 'out_h')
        return self

    def register_results_in_gis(self):
        """Register the generated maps in the temporal database
        Loop through output names
        if no output name is provided, don't do anything
        if name is populated, create a strds of the right temporal type
        and register the corresponding listed maps
        """
        for mkey, lst in self.output_maplist.iteritems():
            strds_name = self.out_map_names[mkey]
            if strds_name is None:
                continue
            self.gis.register_maps_in_strds(mkey, strds_name, lst,
                                            self.temporal_type)
        return self


class TimedArray(object):
    """A container for np.ndarray along with time informations
    Update the array value according to the simulation time
    array is accessed via get_array()
    """
    def __init__(self, mkey, igis, f_arr_def):
        assert isinstance(mkey, basestring), "not a string!"
        assert hasattr(f_arr_def, '__call__'), "not a function!"
        self.mkey = mkey
        self.igis = igis  # GIS interface
        # A function to generate a default array
        self.f_arr_def = f_arr_def
        # default values for start and end
        # intended to trigger update when is_valid() is first called
        self.a_start = datetime(1, 1, 2)
        self.a_end = datetime(1, 1, 1)

    def get_array(self, sim_time):
        """Return a numpy array valid for the given time
        If the array stored is not valid, update the values of the object
        """
        assert isinstance(sim_time, datetime), "not a datetime object!"
        if not self.is_valid(sim_time):
            self.update_values_from_gis(sim_time)
        return self.arr

    def is_valid(self, sim_time):
        """input being a time in timedelta
        If the current stored array is within the range of the map,
        return True
        If not return False
        """
        if self.a_start <= sim_time <= self.a_end:
            return True
        else:
            return False

    def update_values_from_gis(self, sim_time):
        """Update array, start_time and end_time from GIS
        if GIS return None, set array to default value
        """
        # Retrieve values
        arr, arr_start, arr_end = self.igis.get_array(self.mkey, sim_time)
        # set to default if no array retrieved
        if not isinstance(arr, np.ndarray):
            arr = self.f_arr_def()
        # check retrieved values
        assert isinstance(arr_start, datetime), "not a datetime object!"
        assert isinstance(arr_end, datetime), "not a datetime object!"
        assert arr_start <= sim_time <= arr_end, "wrong time retrieved!"
        # update object values
        self.a_start = arr_start
        self.a_end = arr_end
        self.arr = arr
        return self


class MassBal(object):
    """Follow-up the mass balance during the simulation run
    Mass balance error is the difference between the actual volume and
    the theoretical volume. The later is the old volume + input - output.
    Intended use:
    at each time-step, using add_value():
    individual simulation operations send relevant values to the MassBal object
    at each record time, using write_values():
    averaged or cumulated values for the considered time difference are
    written to a CSV file
    """
    def __init__(self, file_name='', dom_size=0):
        self.dom_size = dom_size
        # values to be written on each record time
        self.fields = ['sim_time',  # either seconds or datetime
                       'avg_timestep', '#timesteps',
                       'boundary_vol', 'rain_vol', 'inf_vol',
                       'inflow_vol', 'hfix_vol',
                       'domain_vol', 'vol_error', '%error',
                       'comp_duration', 'avg_cell_per_sec']
        # data written to file as one line
        self.line = dict.fromkeys(self.fields)
        # data collected during simulation
        self.sim_data = {'tstep': [], 'boundary_vol': [],
                         'rain_vol': [], 'inf_vol': [], 'inflow_vol': [],
                         'old_dom_vol': [], 'new_dom_vol': [],
                         'step_duration': [], 'hfix_vol': []}
        # set file name and create file
        self.file_name = self.set_file_name(file_name)
        self.create_file()

    def set_file_name(self, file_name):
        '''Generate output file name
        '''
        if not file_name:
            file_name = "{}_stats.csv".format(
                str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
        return file_name

    def create_file(self):
        '''Create a csv file and write headers
        '''
        with open(self.file_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
        return self

    def add_value(self, key, value):
        '''add a value to sim_data
        '''
        assert key in self.sim_data, "unknown key!"
        self.sim_data[key].append(value)
        return self

    def write_values(self, sim_time):
        '''Calculate statistics and write them to the file
        '''
        # check if all elements have the same number of records
        rec_len = [len(l) for l in self.sim_data.values()]
        assert rec_len[1:] == rec_len[:-1], "inconsistent number of records!"

        self.line['sim_time'] = sim_time
        # number of time-step during the interval is the number of records
        self.line['#timesteps'] = len(self.sim_data['tstep'])
        # average time-step calculation
        elapsed_time = sum(self.sim_data['tstep'])
        avg_timestep = elapsed_time / self.line['#timesteps']
        self.line['avg_timestep'] = '{:.3f}'.format(avg_timestep)

        # sum of inflow (positive) / outflow (negative) volumes
        boundary_vol = sum(self.sim_data['boundary_vol'])
        self.line['boundary_vol'] = '{:.3f}'.format(boundary_vol)
        rain_vol = sum(self.sim_data['rain_vol'])
        self.line['rain_vol'] = '{:.3f}'.format(rain_vol)
        inf_vol = - sum(self.sim_data['inf_vol'])
        self.line['inf_vol'] = '{:.3f}'.format(inf_vol)
        inflow_vol = sum(self.sim_data['inflow_vol'])
        self.line['inflow_vol'] = '{:.3f}'.format(inflow_vol)
        hfix_vol = sum(self.sim_data['hfix_vol'])
        self.line['hfix_vol'] = '{:.3f}'.format(hfix_vol)

        # For domain volume, take last value(i.e. current)
        last_vol = self.sim_data['new_dom_vol'][-1]
        self.line['domain_vol'] = '{:.3f}'.format(last_vol)

        # mass error is the diff. between the theor. vol and the actual vol
        first_vol = self.sim_data['old_dom_vol'][0]
        sum_ext_vol = sum([boundary_vol, rain_vol, inf_vol,
                           inflow_vol, hfix_vol])
        dom_vol_theor = first_vol + sum_ext_vol
        vol_error = last_vol - dom_vol_theor
        self.line['vol_error'] = '{:.3f}'.format(vol_error)
        if last_vol <= 0:
            self.line['%error'] = '-'
        else:
            self.line['%error'] = '{:.2%}'.format(vol_error / last_vol)

        # Performance
        comp_duration = sum(self.sim_data['step_duration'])
        self.line['comp_duration'] = '{:.3f}'.format(comp_duration)
        # Average step computation time
        avg_comp_time = comp_duration / rec_len[0]
        self.line['avg_cell_per_sec'] = int(self.dom_size / avg_comp_time)

        # Add line to file
        with open(self.file_name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(self.line)

        # empty dictionaries
        self.sim_data = {k: [] for k in self.sim_data.keys()}
        self.line = dict.fromkeys(self.line.keys())
        return self
