
Configuration file
==================

The parameters of a simulation are given through a configuration file in
a format similar to Microsoft Windows INI files.
An example is given in the tutorial above.
The file is separated in sections described below.

[time]
------

Simulation duration could be given by a combination of start time, end
time and duration. If only the duration is given, the results will be
written as a relative time STRDS. In case start time is given, the
simulation will use an absolute temporal type.

+----------------+------------------------------------------------------+--------------------+
| Keyword        | Description                                          | Format             |
+================+======================================================+====================+
| start_time     | Starting time                                        | yyyy-mm-dd HH:MM   |
+----------------+------------------------------------------------------+--------------------+
| end_time       | Ending time                                          | yyyy-mm-dd HH:MM   |
+----------------+------------------------------------------------------+--------------------+
| duration       | Simulation duration                                  | HH:MM:SS           |
+----------------+------------------------------------------------------+--------------------+
| record_step    | Time-step at which results are written to the disk   | HH:MM:SS           |
+----------------+------------------------------------------------------+--------------------+

Valid combinations:

-  *start_time* and *end_time*
-  *start_time* and *duration*
-  *duration* only

[input]
-------

Itzï does not support Lat-Long coordinates. A projected location should
be used. The inputs maps could be given either as STRDS or single maps.
First, the module try to load a STRDS of the given name. If
unsuccessful, it will load the given map, and stop with an error if the
name does not correspond to either a map or a STRDS.

The following inputs are mandatory:

-  Digital elevation model in meters
-  Friction, expressed as Manning's *n*

.. list-table::
   :header-rows: 1
   :widths: 25 60 15

   * - Keyword
     - Description
     - Unit
   * - dem
     - Terrain elevation.
     - m
   * - water_depth
     - Starting water depth.
     - m
   * - water_surface_elevation
     - Starting water surface elevation.
       Equivalent to *dem* + *water_depth*.
     - m
   * - friction
     - Friction as Manning’s *n*
     - m s-(1/3)
   * - rain
     - Rainfall rate.
     - mm/h
   * - inflow
     - Point inflow.
       (ex: for 20 m3/s on a 10x10 cell, velocity is 0.2 m/s)
     - m/s
   * - bctype
     - Boundary conditions type.
     - None
   * - bcval
     - Boundary conditions values.
     - m
   * - infiltration
     - User-defined infiltration rate.
     - mm/h
   * - effective_porosity
     - Effective porosity. Used for Green-Ampt infiltration.
     - None
   * - capillary_pressure
     - Wetting front capillary pressure head. Also called suction head.
       Used for Green-Ampt infiltration.
     - mm
   * - hydraulic_conductivity
     - Soil hydraulic conductivity. Used for Green-Ampt infiltration.
     - mm/h
   * - soil_water_content
     - Relative soil water content. Used for Green-Ampt infiltration.
       Available porosity is *effective_porosity* - *soil_water_content*.
     - None
   * - losses
     - User-defined loss rate.
     - mm/h

.. versionchanged:: 25.8
    *start_h* to *water_depth*.

.. versionadded:: 25.8
    *soil_water_content* and *water_surface_elevation*.

.. note:: When using a deprecated keyword, a warning will be displayed, and the map loaded normally.
    However, you must update you input file, as the deprecation warning will go away in the future, and the unrecognized name will be ignored.

Every input could be either a map or a Space-Time Raster Dataset.

If the selected input are located in another GRASS mapset than the current one (or the one specified in the [grass] section),
you must define the full map ID (map\@mapset) and add those mapsets to the GRASS search path with ``g.mapsets``.

Boundary conditions
^^^^^^^^^^^^^^^^^^^

Boundary conditions type are defined by an integer:

-  0 or 1: Closed boundary (default)
-  2: Open boundary: velocity at the boundary is equal to the velocity
   inside the domain
-  3: Not implemented yet
-  4: User-defined water depth inside the domain

The "open" and "closed" boundary conditions are applied only at the border of the GRASS computational region.

Infiltration
^^^^^^^^^^^^

Two infiltration models are available:

-  A user-defined rate, set with *infiltration*
-  The Green-Ampt, model, by setting *effective_porosity*, *capillary_pressure* and *hydraulic_conductivity*. *soil_water_content* can also be set.

*infiltration* and any of the Green-Ampt parameters are mutually exclusives.
Likewise, if any of *effective_porosity*, *capillary_pressure* and *hydraulic_conductivity* is given, all the others should be given as well.

.. caution:: Although all inputs could vary in time, allowing some to do so might result in unexpected behaviour.
    For example, time-varying *dem* is possible but has never been tested.
    Also, forcing a change in water depth in time with either *water_depth* or *water_surface_elevation* will conflict with the internal water depth computation.

[output]
--------

+-----------+------------------------------------------------+------------------------+
| Keyword   | Description                                    | Format                 |
+===========+================================================+========================+
| prefix    | Prefix of output STRDS                         | string                 |
+-----------+------------------------------------------------+------------------------+
| values    | Values to be saved. Each one will be a STRDS   | comma separated list   |
+-----------+------------------------------------------------+------------------------+

The possible values to be exported are the following:

+-------------------------+---------------------------------------------------------+--------+
| Keyword                 | Description                                             | Unit   |
+=========================+=========================================================+========+
| water_depth             | Water depth                                             | m      |
+-------------------------+---------------------------------------------------------+--------+
| water_surface_elevation | Water surface elevation (depth + elevation)             | m      |
+-------------------------+---------------------------------------------------------+--------+
| v                       | Overland flow speed (velocity's magnitude)              | m/s    |
+-------------------------+---------------------------------------------------------+--------+
| vdir                    | Velocity's direction. Counter-clockwise from East       | degrees|
+-------------------------+---------------------------------------------------------+--------+
| froude                  | The Froude number                                       | none   |
+-------------------------+---------------------------------------------------------+--------+
| qx                      | Volumetric flow, x direction. Positive if going East    | m³/s   |
+-------------------------+---------------------------------------------------------+--------+
| qy                      | Volumetric flow, y direction. Positive if going South   | m³/s   |
+-------------------------+---------------------------------------------------------+--------+
| mean_boundary_flow      | Flow coming in (positive) or going out (negative) the   | m/s    |
|                         | domain due to boundary conditions. Mean since the       |        |
|                         | last record                                             |        |
+-------------------------+---------------------------------------------------------+--------+
| mean_infiltration       | Mean infiltration rate since the last record            | mm/h   |
+-------------------------+---------------------------------------------------------+--------+
| mean_rainfall           | Mean rainfall rate since the last record                | mm/h   |
+-------------------------+---------------------------------------------------------+--------+
| mean_inflow             | Mean user flow since the last record                    | m/s    |
+-------------------------+---------------------------------------------------------+--------+
| mean_losses             | Mean losses since the last record                       | mm/h   |
+-------------------------+---------------------------------------------------------+--------+
| mean_drainage_flow      | Mean exchange flow between surface and drainage model   |        |
|                         | since the last record                                   | m/s    |
+-------------------------+---------------------------------------------------------+--------+
| volume_error            | Total created volume due to numerical error since the   | m³     |
|                         | last record                                             |        |
+-------------------------+---------------------------------------------------------+--------+

.. versionchanged:: 25.8
    *h* to *water_depth*.
    *wse* to *water_surface_elevation*.
    *boundaries* to *mean_boundary_flow*.
    *verror* to *volume_error*.
    *inflow* to *mean_inflow*.
    *infiltration* to *mean_infiltration*.
    *rainfall* to *mean_rainfall*.
    *losses* to *mean_losses*.
    *drainage_stats* to *mean_drainage_flow*.

.. versionchanged:: 25.8
  For coherence with the input unit, *mean_losses* is in mm/h instead of m/s.

.. versionadded:: 25.8
    *froude*.

.. caution:: If a deprecated output name is requested, a warning will be displayed and the new, correct output will be written to disk.
    You must update your configuration file, as the deprecation substitution will be removed in a future version.

In addition to output a map at each *record_step*, *water_depth* and *v* also
produce each a map of maximum values attained all over the domain since the beginning of the simulation.

In the water depth maps, the values under the *hmin* threshold are masked with the *r.null* GRASS command.
This does not apply to the map of maximum values.

If an exported map is totally empty, it is deleted at the end of the simulation when registered in the STRDS.

[statistics]
------------

+---------------+-------------------+-------------+
| Keyword       | Description       | Format      |
+===============+===================+=============+
| stats_file    | Statistics file   | CSV table   |
+---------------+-------------------+-------------+

Statistics file
^^^^^^^^^^^^^^^

The statistic file is a CSV file updated at each *record_step*.
The values exported are shown in the table below.
Water entering the domain is represented by a positive value.
Water leaving the domain is negative.

+-------------------------+----------------------------------------------------------------------+--------+
| Keyword                 | Description                                                          | Unit   |
+=========================+======================================================================+========+
| simulation_time         | Total elapsed simulation time.                                       | time   |
+-------------------------+----------------------------------------------------------------------+--------+
| average_timestep        | Average time-step duration since last record.                        | s      |
+-------------------------+----------------------------------------------------------------------+--------+
| timesteps               | Number of time-steps since the last record.                          | none   |
+-------------------------+----------------------------------------------------------------------+--------+
| boundary_volume         | Water volume that passed the domain boundaries since last record.    | m³     |
+-------------------------+----------------------------------------------------------------------+--------+
| rainfall_volume         | Rain volume that entered the domain since last record.               | m³     |
+-------------------------+----------------------------------------------------------------------+--------+
| infiltration_volume     | Water volume that left the domain due to infiltration since          | m³     |
|                         | last record.                                                         |        |
+-------------------------+----------------------------------------------------------------------+--------+
| inflow_volume           | Water volume that entered or left the domain due to user             | m³     |
|                         | inflow since last record.                                            |        |
+-------------------------+----------------------------------------------------------------------+--------+
| losses_volume           | Water volume that entered or left the domain due to                  | m³     |
|                         | losses since last record.                                            |        |
+-------------------------+----------------------------------------------------------------------+--------+
| drainage_network_volume | Water volume that entered or left the surface domain since           | m³     |
|                         | last record due to exchanges with the drainage network.              |        |
+-------------------------+----------------------------------------------------------------------+--------+
| domain_volume           | Total water volume in the domain at this time-step.                  | m³     |
+-------------------------+----------------------------------------------------------------------+--------+
| volume_change           | Changes in volume since the last record.                             | m³     |
+-------------------------+----------------------------------------------------------------------+--------+
| volume_error            | Water volume created due to numerical errors since last record.      | m³     |
+-------------------------+----------------------------------------------------------------------+--------+
| percent_error           | Percentage of the domain volume change due to numerical              | %      |
|                         | error. Corresponds to *volume_error* / *volume_change* \* 100        |        |
+-------------------------+----------------------------------------------------------------------+--------+

*volume_change* is equal to the sum of *boundary_volume*, *rainfall_volume*, *infiltration_volume*, *inflow_volume*, *losses_volume*, *drainage_network_volume*, and *volume_error*.
However, due to the way the volumes are computed internally, small variations could occur.

.. versionchanged:: 25.8
    Columns names are more explicit. *volume_change* is added.


[options]
---------

.. versionadded:: 25.8
    *max_error* is added.

+----------+----------------------------------------------+----------------+---------------+
| Keyword  | Description                                  | Format         | Default value |
+==========+==============================================+================+===============+
| hmin     | Water depth threshold in metres              | positive float | 0.005         |
+----------+----------------------------------------------+----------------+---------------+
| cfl      | Coefficient applied to calculate time-step   | positive float | 0.7           |
+----------+----------------------------------------------+----------------+---------------+
| theta    | Inertia weighting coefficient                | float between  | 0.9           |
|          |                                              | 0 and 1        |               |
+----------+----------------------------------------------+----------------+---------------+
| vrouting | Routing velocity in m/s                      | positive float | 0.1           |
+----------+----------------------------------------------+----------------+---------------+
| dtmax    | Maximum surface flow time-step in seconds.   | positive float | 5.0           |
+----------+----------------------------------------------+----------------+---------------+
| dtinf    | Time-step of infiltration and losses, in s   | positive float | 60.0          |
+----------+----------------------------------------------+----------------+---------------+
| max_error| Maximum relative volume error.               | positive float | 0.05          |
|          | Simulation will stop if above.               |                |               |
+----------+----------------------------------------------+----------------+---------------+

When water depth is under *hmin*, the flow is routed at the fixed velocity defined by *vrouting*.


[drainage]
----------

This section is needed only if carrying out a simulation that couples drainage and surface flow.

.. warning:: This functionality is still new and in need of testing.
    Use with care.

+---------------------+------------------------------------------------------------+---------------+
| Keyword             | Description                                                | Default value |
+=====================+============================================================+===============+
| swmm_inp            | Path to the EPA SWMM configuration file (.inp)             |               |
+---------------------+------------------------------------------------------------+---------------+
| output              | Name of the output Space Time Vector Dataset where         |               |
|                     | are written the results of the drainage network simulation |               |
+---------------------+------------------------------------------------------------+---------------+
| orifice_coeff       | Orifice coefficient for calculating the flow exchange      | 0.167         |
+---------------------+------------------------------------------------------------+---------------+
| free_weir_coeff     | Free weir coefficient for calculating the flow exchange    | 0.54          |
+---------------------+------------------------------------------------------------+---------------+
| submerged_weir_coeff| Submerged weir coefficient for flow exchange calculation   | 0.056         |
+---------------------+------------------------------------------------------------+---------------+


Drainage output
^^^^^^^^^^^^^^^

The results from the drainage network simulation are saved as vector maps, organised in two layers.
The nodes are stored in layer 1, the links in layer 2.

The values stored for the nodes are described below. All are instantaneous.

.. versionchanged:: 25.8
    Tables columns names are more explicit.

+------------------+-----------------------------------------------------------------------+
| Column           | Description                                                           |
+==================+=======================================================================+
| cat              | DB key                                                                |
+------------------+-----------------------------------------------------------------------+
| node_id          | Name of the node                                                      |
+------------------+-----------------------------------------------------------------------+
| node_type        | Node type  (junction, storage, outlet etc.)                           |
+------------------+-----------------------------------------------------------------------+
| coupling_type    | Equation used for the drainage/surface linkage                        |
+------------------+-----------------------------------------------------------------------+
| coupling_flow    | Flow moving from the drainage to the surface                          |
+------------------+-----------------------------------------------------------------------+
| inflow           | Flow entering the node (m³/s)                                         |
+------------------+-----------------------------------------------------------------------+
| outflow          | Flow exiting the node (m³/s)                                          |
+------------------+-----------------------------------------------------------------------+
| lateral_inflow   | SWMM lateral flow (m³/s)                                              |
+------------------+-----------------------------------------------------------------------+
| losses           | Losses Rate (evaporation and exfiltration).                           |
+------------------+-----------------------------------------------------------------------+
| overflow         | Losses due to node overflow                                           |
+------------------+-----------------------------------------------------------------------+
| depth            | Water depth in m                                                      |
+------------------+-----------------------------------------------------------------------+
| head             | Hydraulic head in metre                                               |
+------------------+-----------------------------------------------------------------------+
| crest_elevation  | Elevation of the top of the node in metres                            |
+------------------+-----------------------------------------------------------------------+
| invert_elevation | Elevation of the bottom of the node in metres                         |
+------------------+-----------------------------------------------------------------------+
| initial_depth    | Water depth in the node at the start of the simulation                |
+------------------+-----------------------------------------------------------------------+
| full_depth       | *crest_elevation* - *invert_elevation* (m)                            |
+------------------+-----------------------------------------------------------------------+
| surcharge_depth  | Depth above *crest_elevation* before overflow begins                  |
+------------------+-----------------------------------------------------------------------+
| ponding_area     | Area above the node where ponding occurs (m²)                         |
+------------------+-----------------------------------------------------------------------+
| volume           | Water volume in the node                                              |
+------------------+-----------------------------------------------------------------------+
| full_volume      | Volume in the node when *head - invert_elevation = crest_elevation*   |
+------------------+-----------------------------------------------------------------------+

The values for the links are as follows:

+---------------+-------------------------------------------------------+
| Column        | Description                                           |
+===============+=======================================================+
| cat           | DB key                                                |
+---------------+-------------------------------------------------------+
| link_id       | Name of the link                                      |
+---------------+-------------------------------------------------------+
| link_type     | Link type (conduit, pump etc.)                        |
+---------------+-------------------------------------------------------+
| flow          | Volumetric flow (m³/s)                                |
+---------------+-------------------------------------------------------+
| depth         | Water depth in the conduit (m)                        |
+---------------+-------------------------------------------------------+
| volume        | Water volume stored in the conduit (m³)               |
+---------------+-------------------------------------------------------+
| inlet_offset  | Height above inlet node invert elevation (m)          |
+---------------+-------------------------------------------------------+
| outlet_offset | Height above outlet node invert elevation (m)         |
+---------------+-------------------------------------------------------+
| froude        | Average Froude number                                 |
+---------------+-------------------------------------------------------+

.. note:: Only links and nodes with coordinates will be written as geographic features to the grass vector map.
  However, results from all nodes and links are written to the database, even without an associated geographic feature.

[grass]
-------

Setting those parameters allows to run simulation outside the GRASS shell.
This is especially useful for batch processing involving different locations and mapsets.
If Itzï is run from within the GRASS shell, this section is not necessary.

+--------------+---------------------------------------------+---------+
| Keyword      | Description                                 | Format  |
+==============+=============================================+=========+
| grass_bin    | Path to the grass binary                    | string  |
+--------------+---------------------------------------------+---------+
| grassdata    | Full path to the GIS DataBase               | string  |
+--------------+---------------------------------------------+---------+
| location     | Name of the location                        | string  |
+--------------+---------------------------------------------+---------+
| mapset       | Name of the mapset                          | string  |
+--------------+---------------------------------------------+---------+
| region       | Name of region setting                      | string  |
+--------------+---------------------------------------------+---------+
| mask         | Name of the raster map to be used as a mask | string  |
+--------------+---------------------------------------------+---------+

With GNU/Linux, *grass_bin* could simply be ``grass``.

The *region* and *mask* parameters are optionals and are applied only during the simulation.
After the simulation, those parameters are returned to the previous *region* and *mask* setting.
