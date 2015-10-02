#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE:    t.sim.flood

AUTHOR(S): Laurent Courty

PURPOSE:   Simulate superficial water flows using a quasi-2D implementation
           of the Shallow Water Equations.
           See:
           De Almeida, G. & Bates, P., 2013. Applicability of the local
           inertial approximation of the shallow water equations to
           flood modeling. Water Resources Research, 49(8), pp.4833–4844.
           Sampson, C.C. et al., 2013. An automated routing methodology
           to enable direct rainfall in high resolution shallow water models.
           Hydrological Processes, 27(3), pp.467–476.

COPYRIGHT: (C) 2015 by Laurent Courty

            This program is free software; you can redistribute it and/or
            modify it under the terms of the GNU General Public License
            as published by the Free Software Foundation; either version 2
            of the License, or (at your option) any later version.

            This program is distributed in the hope that it will be useful,
            but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
            GNU General Public License for more details.
"""

#%module
#% description: Simulate superficial flows using simplified shallow water equations
#% keywords: raster
#% keywords: Shallow Water Equations
#% keywords: flow
#% keywords: flood
#%end

#%option G_OPT_R_ELEV
#% key: in_z
#% description: Name of input elevation raster map
#% required: yes
#%end

#%option G_OPT_R_INPUT
#% key: in_n
#% description: Name of input friction coefficient raster map
#% required: yes
#%end

#%option G_OPT_R_INPUT
#% key: in_h
#% description: Name of input water depth raster map
#% required: no
#%end

#~ #%option G_OPT_R_INPUT
#~ #% key: in_y
#~ #% description: Name of input water surface elevation raster map
#~ #% required: no
#~ #%end

#~ #%option G_OPT_R_INPUT
#~ #% key: in_inf
#~ #% description: Name of input infiltration raster map
#~ #% required: no
#~ #%end

#%option G_OPT_STRDS_INPUT
#% key: in_rain
#% description: Name of input rainfall raster space-time dataset
#% required: no
#%end

#%option G_OPT_STRDS_INPUT
#% key: in_inflow
#% description: Name of input user flow raster space-time dataset
#% required: no
#%end

#%option G_OPT_R_INPUT
#% key: in_bc
#% description: Name of input boundary conditions type map
#% required: no
#%end

#%option G_OPT_STRDS_INPUT
#% key: in_bcval
#% description: Name of input boundary conditions values raster STDS
#% required: no
#%end


#%option G_OPT_STRDS_OUTPUT
#% key: out_h
#% description: Name of output water depth raster space-time dataset
#% required: no
#%end

#%option G_OPT_STRDS_OUTPUT
#% key: out_wse
#% description: Name of output water surface elevation space-time dataset
#% required: no
#%end

#~ #%option G_OPT_R_OUTPUT
#~ #% key: q_i
#~ #% description: Name of output flow raster map for x direction
#~ #% required: no
#~ #%end
#~ 
#~ #%option G_OPT_R_OUTPUT
#~ #% key: q_j
#~ #% description: Name of output flow raster map for y direction
#~ #% required: no
#~ #%end

#%option G_OPT_UNDEFINED
#% key: start_time
#% description: Start of the simulation, format yyyy-mm-dd HH:MM
#% required: no
#%end

#%option G_OPT_UNDEFINED
#% key: end_time
#% description: End of the simulation, format yyyy-mm-dd HH:MM
#% required: no
#%end

#%option G_OPT_UNDEFINED
#% key: sim_duration
#% description: Duration of the simulation, format HH:MM:SS
#% required: no
#%end

#%option G_OPT_UNDEFINED
#% key: record_step
#% description: Duration between two records, format HH:MM:SS
#% required: yes
#%end

import sys
import os
from datetime import datetime
import numpy as np
import cProfile
import pstats
import StringIO

import grass.script as grass
from grass.pygrass.gis.region import Region
from grass.pygrass.messages import Messenger

import simulation


def main():
    # start profiler
    pr = cProfile.Profile()
    pr.enable()

    # start messenger
    msgr = Messenger()

    # values to be passed to simulation
    input_times = {'start':None,'end':None,'duration':None,'rec_step':None}
    input_map_names = {'z': None, 'n': None, 'h_old': None,
                'rain': None, 'inf':None, 'bcval': None, 'bctype': None}
    output_map_names = {'out_h':None, 'out_wse':None,
                'out_vx':None, 'out_vy':None, 'out_qx':None, 'out_qy':None}
    temporal_type = ''

    # check and load input values
    temporal_type = read_input_time(options, input_times)
    assert temporal_type in ('relative', 'absolute')
    read_maps_names(options, input_map_names, output_map_names)

    # Run simulation
    sim = simulation.SuperficialFlowSimulation(
                        start_time=input_times['start'],
                        end_time=input_times['end'],
                        sim_duration=input_times['duration'],
                        record_step=input_times['rec_step'],
                        input_maps=input_map_names,
                        output_maps=output_map_names)
    sim.run()

    # end profiling
    pr.disable()
    stat_stream = StringIO.StringIO()
    sortby = 'time'
    ps = pstats.Stats(pr, stream=stat_stream).sort_stats(sortby)
    ps.print_stats(5)
    print stat_stream.getvalue()

def str_to_timedelta(inp_str):
    """Takes a string in the form HH:MM:SS
    and return a timedelta object
    """
    data = inp_str.split(":")
    hours = int(data[0])
    minutes = int(data[1])
    seconds = int(data[2])
    if hours < 0:
        raise ValueError
    if not 0 < minutes < 59 or not 0 < seconds < 59:
        raise ValueError
    obj_dt = timedelta(hours=hours,
                    minutes=minutes,
                    seconds=seconds)
    return obj_dt

def read_input_time(opts, input_times):
    """Check the sanity of input time information
    write the value to relevant dict
    """
    # Default temporal type
    temporal_type = "absolute"

    date_format = '%Y-%m-%d %H:%M'
    # record step
    try:
        input_times['rec_step'] = str_to_timedelta(opts['record_step'])
    except ValueError:
        msgr.fatal(_("{}: format should be HH:MM:SS".format(
                'record_step')))

    # check valid combination to get simulation duration
    b_dur = (opts['sim_duration']
                and not opts['start_time'] and not opts['end_time'])
    b_start_dur = (opts['start_time'] and opts['sim_duration']
                and not opts['end_time'])
    b_start_end = (opts['start_time'] and opts['end_time']
                and not opts['sim_duration'])
    if not (b_dur or b_start_dur or b_start_end):
        msgr.fatal(_(
        "accepted combinations: {d} alone, {s} and {d}, {s} and {e}").format(
                    d='sim_duration', s='start_time', e='end_time'))

    if opts['sim_duration']:
        try:
            input_times['duration'] = str_to_timedelta(opts['sim_duration'])
        except ValueError:
            msgr.fatal(_("{}: format should be HH:MM:SS".format(
                    'sim_duration')))

    if options['end_time']:
        try:
            input_times['end'] = datetime.strptime(opts['end_time'], date_format)
        except ValueError:
            msgr.fatal(_("{}: format should be yyyy-mm-dd HH:MM".format(
                        'end_time')))

    if opts['start_time']:
        try:
            input_times['start'] = datetime.strptime(opts['start_time'],
                                                    date_format)
        except ValueError:
            msgr.fatal(_("{}: format should be yyyy-mm-dd HH:MM".format(
                    'start_time')))
    else:
        input_times['start'] = datetime.min
        temporal_type = "relative"

    return temporal_type

def read_maps_names(opts, input_map_names, output_map_names):
    """
    """
    input_map_names['z'] = opts['in_z']
    input_map_names['n'] = opts['in_n']
    input_map_names['h_old'] = opts['in_h']
    input_map_names['rain'] = opts['in_rain']
    input_map_names['inf'] = opts['in_inflow']
    input_map_names['bctype'] = opts['in_bc']
    input_map_names['bcval'] = opts['in_bcval']

    out_names = {k:v for k,v in opt.items() if k in output_map_names}
    for k in out_names:
        assert k in output_map_names, "wrong list comprehension"
        pass
    output_map_names.update(out_names)


############
# old code #
############
def old_code():

    ###############
    # output data #
    ###############
    # list of written maps:
    list_h = []
    list_wse = []

    #####################################
    # Create output space-time datasets #
    #####################################
    stds_h_id, stds_wse_id = stds.create_stds(
        mapset, options['out_h'], options['out_wse'],
        sim_start_time, can_ovr)

    #####################
    # START COMPUTATION #
    #####################
    # time-step counter
    Dt_c = 0

    # Start time-stepping
    while not domain.sim_clock >= domain.end_time:
        #########################
        # write simulation data #
        #########################
        if domain.sim_clock / record_t >= record_count:
            list_h, list_wse = rw.write_sim_data(
                options['out_h'], options['out_wse'], domain.arr_h_np1,
                domain.arr_z, can_ovr, domain.sim_clock, list_h, list_wse)
            # update record count
            record_count += 1
            # set next forced timestep
            domain.set_forced_timestep(record_count * record_t)


        ############################
        # mass balance calculation #
        ############################
        # calculate and display total grid volume
        domain.solve_gridvolume()
        grass.verbose(_("Domain volume at time %.1f : %.3f ") %
                        (round(domain.sim_clock,1), round(domain.grid_volume,3)))
        # calculate grid volume change
        Dvol = (np.sum(domain.arr_h_np1) - np.sum(domain.arr_h)) * domain.cell_surf
        domain.solve_ext_volume(bound_vol)
        ext_input = domain.total_ext_volume
        # calculate mass balance
        mass_balance = bound_vol + ext_input - Dvol
        # display mass balance
        grass.verbose(_("Mass balance at time %.1f : %.3f ") %
                        (round(domain.sim_clock, 1), round(mass_balance, 3)))

    ##############################
    # write last simulation data #
    ##############################
    
    list_h, list_wse = rw.write_sim_data(
        options['out_h'], options['out_wse'],
        domain.arr_h_np1, domain.arr_z,
        can_ovr, domain.sim_clock, list_h, list_wse)

    # print grid volume
    #~ V_total = np.sum(depth_grid) * domain.cell_surf
    #~ grass.info(_("Total grid volume at time %.1f : %.3f ") %
                #~ (round(sim_clock,1), round(V_total,3)))

    ########################################
    # register maps in space-time datasets #
    ########################################
    # depth
    if options['out_h']:
        list_h = ','.join(list_h) # transform map list into a string
        kwargs = {'maps': list_h,
                'start': 0,
                'unit':'seconds',
                'increment':int(record_t)}
        tgis.register.register_maps_in_space_time_dataset('rast',
                                                stds_h_id, **kwargs)
    # water surface elevation
    if options['out_wse']:
        list_wse = ','.join(list_wse)  # transform map list into a string
        kwargs = {'maps': list_wse,
                'start': 0,
                'unit':'seconds',
                'increment':int(record_t)}
        tgis.register.register_maps_in_space_time_dataset('rast',
                                                stds_wse_id, **kwargs)

    return 0


if __name__ == "__main__":
    options, flags = grass.parser()
    sys.exit(main())
