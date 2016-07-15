# coding=utf8
"""
Copyright (C) 2015-2016 Laurent Courty

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
import ConfigParser
from datetime import datetime, timedelta
import numpy as np

import grass.script as grass

import gis


class ConfigReader(object):
    """Parse the configuration file and check validity of given options
    """
    def __init__(self, filename, msgr):
        self.msgr = msgr
        self.config_file = filename
        # default values to be passed to simulation
        self.__set_defaults()
        # read entry values
        self.set_entry_values()

    def __set_defaults(self):
        """Set dictionaries of default values to be passed to simulation"""
        k_raw_input_times = ['start_time', 'end_time',
                             'duration', 'record_step']
        self.ga_list = ['effective_pororosity', 'capillary_pressure',
                        'hydraulic_conductivity']
        k_input_map_names = ['dem', 'friction', 'start_h', 'start_y',
                             'rain', 'inflow', 'bcval', 'bctype',
                             'infiltration'] + self.ga_list
        k_output_map_names = ['h', 'wse', 'v', 'vdir', 'qx', 'qy',
                              'boundaries', 'infiltration', 'rainfall',
                              'inflow', 'drainage']
        self.sim_param = {'hmin': 0.005, 'cfl': 0.7, 'theta': 0.9,
                          'g': 9.80665, 'vrouting': 0.1, 'dtmax': 5.,
                          'slmax': .1, 'dtinf': 60., 'inf_model': None}
        self.raw_input_times = dict.fromkeys(k_raw_input_times)
        self.output_map_names = dict.fromkeys(k_output_map_names)
        self.input_map_names = dict.fromkeys(k_input_map_names)
        self.out_prefix = 'itzi_results_{}'.format(datetime.now().strftime('%Y%m%dT%H%M%S'))
        return self

    def set_entry_values(self):
        """Read and check entry values
        """
        # read file and populate dictionaries
        self.read_param_file()
        # process inputs times
        self.sim_times = SimulationTimes(self.raw_input_times, self.msgr)
        # check if mandatory parameters are present
        self.check_mandatory()
        # check coherence of infiltrations entries
        self.check_inf_maps()
        # check if the output files do not exist
        self.check_output_files()
        return self

    def read_param_file(self):
        """Read the parameter file and populate the relevant dictionaries
        """
        self.out_values = []
        # read the file
        params = ConfigParser.SafeConfigParser(allow_no_value=True)
        f = params.read(self.config_file)
        if not f:
            self.msgr.fatal(_(u"File <{}> not found".format(self.config_file)))
        # populate dictionaries using loops instead of using update() method
        # in order to not add invalid key
        for k in self.raw_input_times:
            if params.has_option('time', k):
                self.raw_input_times[k] = params.get('time', k)
        for k in self.sim_param:
            if params.has_option('options', k):
                self.sim_param[k] = params.getfloat('options', k)
        for k in self.input_map_names:
            if params.has_option('input', k):
                self.input_map_names[k] = params.get('input', k)
        # statistic file
        if params.has_option('statistics', 'stats_file'):
            self.stats_file = params.get('statistics', 'stats_file')
        # output maps
        if params.has_option('output', 'prefix'):
            self.out_prefix = params.get('output', 'prefix')
        if params.has_option('output', 'values'):
            self.out_values = params.get('output', 'values').split(',')
            self.out_values = [e.strip() for e in self.out_values]
        self.generate_output_name()
        return self

    def generate_output_name(self):
        """Generate the name of the strds
        """
        for v in self.out_values:
            if v in self.output_map_names:
                self.output_map_names[v] = '{}_{}'.format(self.out_prefix, v)
        return self

    def file_exist(self, name):
        """Return True if name is an existing map or stds, False otherwise
        """
        if not name:
            return False
        else:
            _id = gis.Igis.format_id(name)
            return gis.Igis.name_is_map(_id) or gis.Igis.name_is_stds(_id)

    def check_output_files(self):
        """Check if the output files do not exist
        """
        for v in self.output_map_names.itervalues():
            if self.file_exist(v) and not grass.overwrite():
                self.msgr.fatal(_(u"File {} exists and will not be overwritten".format(v)))

    def check_inf_maps(self):
        """check coherence of input infiltration maps
        set infiltration model type
        """
        inf_k = 'infiltration'
        # if at least one Green-Ampt parameters is present
        ga_any = any(self.input_map_names[i] for i in self.ga_list)
        # if all Green-Ampt parameters are present
        ga_all = all(self.input_map_names[i] for i in self.ga_list)
        # verify parameters
        if not self.input_map_names[inf_k] and not ga_any:
            self.sim_param['inf_model'] = None
        elif self.input_map_names[inf_k] and not ga_any:
            self.sim_param['inf_model'] = 'constant'
        elif self.input_map_names[inf_k] and ga_any:
            self.msgr.fatal(_(u"Infiltration model incompatible with user-defined rate"))
        # check if all maps for Green-Ampt are presents
        elif ga_any and not ga_all:
            self.msgr.fatal(_(u"{} are mutualy inclusive".format(self.ga_list)))
        elif ga_all and not self.input_map_names[inf_k]:
            self.sim_param['inf_model'] = 'green-ampt'
        return self

    def check_mandatory(self):
        """check if mandatory parameters are present
        """
        if not all([self.input_map_names['dem'],
                   self.input_map_names['friction'],
                   self.sim_times.record_step]):
            msgr.fatal(_(u"inputs <dem>, <friction> and <record_step> are mandatory"))

    def display_sim_param(self):
        """Display simulation parameters if verbose
        """
        self.msgr.verbose(u"Using the following parameters:")
        txt_template = u"{}: {}"
        for k, v in self.input_map_names.iteritems():
            self.msgr.verbose(txt_template.format(k, v))
        for k, v in self.output_map_names.iteritems():
            self.msgr.verbose(txt_template.format(k, v))
        for k, v in self.sim_param.iteritems():
            self.msgr.verbose(txt_template.format(k, v))
        # simulation times
        self.msgr.verbose(txt_template.format('start', self.sim_times.start))
        self.msgr.verbose(txt_template.format('end', self.sim_times.end))
        self.msgr.verbose(txt_template.format('duration', self.sim_times.duration))
        self.msgr.verbose(txt_template.format('duration', self.sim_times.duration))


class SimulationTimes(object):
    """Store the information about simulation starting & ending time and duration
    """
    def __init__(self, raw_input_times, msgr):
        self.msgr = msgr
        self.read_simulation_times(raw_input_times)

    def read_simulation_times(self, raw_input_times):
        """Read a given dictionary of input times.
        Check the coherence of the input and store it in the object
        """
        self.raw_duration = raw_input_times['duration']
        self.raw_start = raw_input_times['start_time']
        self.raw_end = raw_input_times['end_time']
        self.raw_record_step = raw_input_times['record_step']

        self.date_format = '%Y-%m-%d %H:%M'
        self.rel_err_msg = u"{}: invalid format (should be HH:MM:SS)"
        self.abs_err_msg = u"{}: invalid format (should be yyyy-mm-dd HH:MM)"

        # check if the given times are coherent
        self.check_combination()

        # transform duration and record_step to timedelta object
        self.duration = self.read_timedelta(self.raw_duration)
        self.record_step = self.read_timedelta(self.raw_record_step)

        # transform start and end to datetime object
        self.start = self.read_datetime(self.raw_start)
        self.end = self.read_datetime(self.raw_end)

        # check coherence of the properties
        self.check_coherence()

        # make sure everything went fine
        assert isinstance(self.end, datetime)
        assert isinstance(self.start, datetime)
        assert isinstance(self.duration, timedelta)
        assert isinstance(self.record_step, timedelta)
        assert self.end >= self.start
        assert self.duration == (self.end - self.start)

        return self

    def str_to_timedelta(self, inp_str):
        """Takes a string in the form HH:MM:SS
        and return a timedelta object
        """
        data = inp_str.split(":")
        hours = int(data[0])
        minutes = int(data[1])
        seconds = int(data[2])
        if hours < 0:
            raise ValueError
        if not 0 <= minutes <= 59 or not 0 <= seconds <= 59:
            raise ValueError
        obj_dt = timedelta(hours=hours,
                           minutes=minutes,
                           seconds=seconds)
        return obj_dt

    def check_combination(self):
        """Verifies if the given input times combination is valid.
        Sets temporal type.
        """
        comb_err_msg = (u"accepted combinations: "
                        u"{d} alone, {s} and {d}, "
                        u"{s} and {e}").format(d='duration',
                                               s='start_time',
                                               e='end_time')
        b_dur = (self.raw_duration and
                 not self.raw_start and
                 not self.raw_end)
        b_start_dur = (self.raw_start and
                       self.raw_duration and
                       not self.raw_end)
        b_start_end = (self.raw_start and self.raw_end and
                       not self.raw_duration)
        if not (b_dur or b_start_dur or b_start_end):
            self.msgr.fatal(_(comb_err_msg))
        # if only duration is given, temporal type is relative
        if b_dur:
            self.temporal_type = 'relative'
        else:
            self.temporal_type = 'absdolute'
        return self

    def read_timedelta(self, string):
        """Try to transform string in timedelta object.
        If it fail, return an error message
        If string is None, return None
        """
        if string:
            try:
                return self.str_to_timedelta(string)
            except ValueError:
                self.msgr.fatal(_(self.rel_err_msg.format(string)))
        else:
            return None

    def read_datetime(self, string):
        """Try to transform string in datetime object.
        If it fail, return an error message
        If string is None, return None
        """
        if string:
            try:
                return datetime.strptime(raw_input_times['end'],
                                         self.date_format)
            except ValueError:
                self.msgr.fatal(_(self.abs_err_msg.format(string)))
        else:
            return None

    def check_coherence(self):
        """Sets end or duration if not given
        Verifies if end is superior to starts
        """
        if self.start is None:
            self.start = datetime.min
        if self.end is None:
            self.end = self.start + self.duration
        if self.start >= self.end:
            self.msgr.fatal(_("Simulation duration must be positive"))
        if self.duration is None:
            self.duration = self.end - self.start
