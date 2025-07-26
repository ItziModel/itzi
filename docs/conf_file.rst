
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
| start\_time    | Starting time                                        | yyyy-mm-dd HH:MM   |
+----------------+------------------------------------------------------+--------------------+
| end\_time      | Ending time                                          | yyyy-mm-dd HH:MM   |
+----------------+------------------------------------------------------+--------------------+
| duration       | Simulation duration                                  | HH:MM:SS           |
+----------------+------------------------------------------------------+--------------------+
| record\_step   | Time-step at which results are written to the disk   | HH:MM:SS           |
+----------------+------------------------------------------------------+--------------------+

Valid combinations:

-  *start\_time* and *end\_time*
-  *start\_time* and *duration*
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

+-------------------------+-----------------------------------------+--------------+
| Keyword                 | Description                             | Format       |
+=========================+=========================================+==============+
| dem                     | Elevation in meters                     | map or strds |
+-------------------------+-----------------------------------------+--------------+
| friction                | Manning's *n* (friction value)          | map or strds |
+-------------------------+-----------------------------------------+--------------+
| water_depth             | Starting water depth in meters          | map or strds |
+-------------------------+-----------------------------------------+--------------+
| rain                    | Rainfall in mm/h                        | map or strds |
+-------------------------+-----------------------------------------+--------------+
| inflow                  | Point inflow in m/s (ex: for 20 m3/s on | map or strds |
|                         | a 10x10 cell, velocity is 0.2 m/s)      |              |
+-------------------------+-----------------------------------------+--------------+
| bctype                  | Boundary conditions type                | map or strds |
+-------------------------+-----------------------------------------+--------------+
| bcval                   | Boundary conditions values              | map or strds |
+-------------------------+-----------------------------------------+--------------+
| infiltration            | Fixed infiltration rate in mm/h         | map or strds |
+-------------------------+-----------------------------------------+--------------+
| effective\_porosity     | Effective porosity in mm/mm             | map or strds |
+-------------------------+-----------------------------------------+--------------+
| capillary\_pressure     | Wetting front capillary pressure head   | map or strds |
|                         | in mm                                   |              |
+-------------------------+-----------------------------------------+--------------+
| hydraulic\_conductivity | Soil hydraulic conductivity in mm/h     | map or strds |
+-------------------------+-----------------------------------------+--------------+
| soil_water_content      | Relative soil water content. in mm/mm   | map or strds |
+-------------------------+-----------------------------------------+--------------+
| losses                  | User-defined losses in mm/h             | map or strds |
|                         | (*new in16.9, renamed in 17.7*)         |              |
+-------------------------+-----------------------------------------+--------------+

.. versionchanged:: 25.7
    *start_h* is renamed *water_depth*.
    *soil_water_content* is added.

.. warning:: If the selected input are located in another GRASS mapset than the current one (or the one specified in the [grass] section),
    you must define the full map ID (map\@mapset) and add those mapsets to the GRASS search path with *g.mapsets*.

Boundary conditions type are defined by an integer as follow:

-  0 or 1: Closed boundary (default)
-  2: Open boundary: velocity at the boundary is equal to the velocity
   inside the domain
-  3: Not implemented yet
-  4: User-defined water depth inside the domain

The "open" and "closed" boundary conditions are applied only at the border of the GRASS computational region.

.. note:: *infiltration* and any of the Green-Ampt parameters are mutually exclusives.
    Likewise, if any of the Green-Ampt parameter is given, all the others should be given as well.

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

+--------------------+---------------------------------------------------------+--------+
| Keyword            | Description                                             | Unit   |
+====================+=========================================================+========+
| water_depth        | Water depth                                             | m      |
+--------------------+---------------------------------------------------------+--------+
| wse                | Water surface elevation (depth + elevation)             | m      |
+--------------------+---------------------------------------------------------+--------+
| v                  | Overland flow speed (velocity's magnitude)              | m/s    |
+--------------------+---------------------------------------------------------+--------+
| vdir               | Velocity's direction. Counter-clockwise from East       | degrees|
+--------------------+---------------------------------------------------------+--------+
| froude             | The Froude number                                       | none   |
+--------------------+---------------------------------------------------------+--------+
| qx                 | Volumetric flow, x direction. Positive if going East    | m³/s   |
+--------------------+---------------------------------------------------------+--------+
| qy                 | Volumetric flow, y direction. Positive if going South   | m³/s   |
+--------------------+---------------------------------------------------------+--------+
| mean_boundary_flow | Flow coming in (positive) or going out (negative) the   | m/s    |
|                    | domain due to boundary conditions. Mean since the       |        |
|                    | last record                                             |        |
+--------------------+---------------------------------------------------------+--------+
| mean_infiltration  | Mean infiltration rate since the last record            | mm/h   |
+--------------------+---------------------------------------------------------+--------+
| mean_rainfall      | Mean rainfall rate since the last record                | mm/h   |
+--------------------+---------------------------------------------------------+--------+
| mean_inflow        | Mean user flow since the last record                    | m/s    |
+--------------------+---------------------------------------------------------+--------+
| mean_losses        | Mean losses since the last record                       | m/s    |
+--------------------+---------------------------------------------------------+--------+
| mean_drainage_flow | Mean exchange flow between surface and drainage model   |        |
|                    | since the last record                                   | m/s    |
+--------------------+---------------------------------------------------------+--------+
| volume_error       | Total created volume due to numerical error since the   | m³     |
|                    | last record                                             |        |
+--------------------+---------------------------------------------------------+--------+


.. versionchanged:: 25.7
    *froude* is added.
    *h* is changed to *water_depth*.
    *boundaries* changed to *mean_boundary_flow*.
    *verror* changed to *volume_error*.
    *inflow* changed to *mean_inflow*.
    *infiltration* changed to *mean_infiltration*.
    *rainfall* changed to *mean_rainfall*.
    *losses* changed to *mean_losses*.
    *drainage_stats* changed to *mean_drainage_flow*.

In addition to output a map at each *record\_step*, *water_depth* and *v* also
produce each a map of maximum values attained all over the domain since the beginning of the simulation.

.. note:: Water depth maps have their values under the *hmin* threshold masked with the ``r.null`` GRASS command.
    This does not apply to the map of maximum values.
    In addition, if an exported map is totally empty, it is deleted at the end of the simulation.

[statistics]
------------

+---------------+-------------------+-------------+
| Keyword       | Description       | Format      |
+===============+===================+=============+
| stats\_file   | Statistics file   | CSV table   |
+---------------+-------------------+-------------+

Statistics file
^^^^^^^^^^^^^^^
The statistic file is a CSV file updated at each *record_step*.
The values exported are shown in the table below.
Water entering the domain is represented by a positive value.
Water leaving the domain is negative.

+-------------------------+------------------------------------------------------------------+--------+
| Keyword                 | Description                                                      | Unit   |
+=========================+==================================================================+========+
| simulation\_time        | Total elapsed simulation time.                                   | time   |
+-------------------------+------------------------------------------------------------------+--------+
| average\_timestep       | Average time-step duration since last record.                    | s      |
+-------------------------+------------------------------------------------------------------+--------+
| timesteps               | Number of time-steps since the last record.                      | none   |
+-------------------------+------------------------------------------------------------------+--------+
| boundary\_volume        | Water volume that passed the domain boundaries since last record.| m³     |
+-------------------------+------------------------------------------------------------------+--------+
| rainfall\_volume        | Rain volume that entered the domain since last record.           | m³     |
+-------------------------+------------------------------------------------------------------+--------+
| infiltration\_volume    | Water volume that left the domain due to infiltration since      | m³     |
|                         | last record.                                                     |        |
+-------------------------+------------------------------------------------------------------+--------+
| inflow\_volume          | Water volume that entered or left the domain due to user         | m³     |
|                         | inflow since last record.                                        |        |
+-------------------------+------------------------------------------------------------------+--------+
| losses\_volume          | Water volume that entered or left the domain due to              | m³     |
|                         | losses since last record.                                        |        |
+-------------------------+------------------------------------------------------------------+--------+
| drainage_network_volume | Water volume that entered or left the surface domain since       | m³     |
|                         | last record due to exchanges with the drainage network.          |        |
+-------------------------+------------------------------------------------------------------+--------+
| domain\_volume          | Total water volume in the domain at this time-step.              | m³     |
+-------------------------+------------------------------------------------------------------+--------+
| volume\_change          | Changes in volume since the last record.                         | m³     |
+-------------------------+------------------------------------------------------------------+--------+
| volume\_error           | Water volume created due to numerical errors since last record.  | m³     |
+-------------------------+------------------------------------------------------------------+--------+
| percent_error           | Percentage of the domain volume change due to numerical          | %      |
|                         | error. Corresponds to *volume\_error* / *volume\_change* \* 100  |        |
+-------------------------+------------------------------------------------------------------+--------+

*volume\_change* is equal to the sum of *boundary\_volume*, *rainfall\_volume*, *infiltration\_volume*, *inflow_volume*, *losses\_volume*, *drainage\_network_volume*, and *volume\_error*.
However, due to the way the volumes are computed internally, small variations could occur.

.. versionchanged:: 25.7
    Columns names are more explicit. *volume_change* is added.


[options]
---------

.. versionadded:: 25.7
    ``max_error`` is added.

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
    It may be buggy. Use with care.

+---------------------+------------------------------------------------------------+---------------+
| Keyword             | Description                                                | Default value |
+=====================+============================================================+===============+
| swmm\_inp           | Path to the EPA SWMM configuration file (.inp)             |               |
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

The output maps are organised in two layers.
The nodes are stored in layer 1, the links in layer 2.

The values stored for the nodes are described below. All are instantaneous.

.. versionchanged:: 25.7
    Tables columns names are more explicit.

+------------------+---------------------------------------------------------------------+
| Column           | Description                                                         |
+==================+=====================================================================+
| cat              | DB key                                                              |
+------------------+---------------------------------------------------------------------+
| node_id          | Name of the node                                                    |
+------------------+---------------------------------------------------------------------+
| node_type        | Node type  (junction, storage, outlet etc.)                         |
+------------------+---------------------------------------------------------------------+
| coupling_type    | Equation used for the drainage/surface linkage                      |
+------------------+---------------------------------------------------------------------+
| coupling_flow    | Flow moving from the drainage to the surface                        |
+------------------+---------------------------------------------------------------------+
| inflow           | Flow entering the node (m³/s)                                       |
+------------------+---------------------------------------------------------------------+
| outflow          | Flow exiting the node (m³/s)                                        |
+------------------+---------------------------------------------------------------------+
| lateral_inflow   | SWMM lateral flow (m³/s)                                            |
+------------------+---------------------------------------------------------------------+
| losses           | Losses Rate (evaporation and exfiltration).                         |
+------------------+---------------------------------------------------------------------+
| overflow         | Losses due to node overflow                                         |
+------------------+---------------------------------------------------------------------+
| depth            | Water depth in m                                                    |
+------------------+---------------------------------------------------------------------+
| head             | Hydraulic head in metre                                             |
+------------------+---------------------------------------------------------------------+
| crest_elevation  | Elevation of the top of the node in metres                          |
+------------------+---------------------------------------------------------------------+
| invert_elevation | Elevation of the bottom of the node in metres                       |
+------------------+---------------------------------------------------------------------+
| initial_depth    | Water depth in the node at the start of the simulation              |
+------------------+---------------------------------------------------------------------+
| full_depth       | *crownElev* - *invertElev* (m)                                      |
+------------------+---------------------------------------------------------------------+
| surcharge_depth  | Depth above *crownElev* before overflow begins                      |
+------------------+---------------------------------------------------------------------+
| ponding_area     | Area above the node where ponding occurs (m²)                       |
+------------------+---------------------------------------------------------------------+
| volume           | Water volume in the node                                            |
+------------------+---------------------------------------------------------------------+
| full_volume      | Volume in the node when *head - invert_elevation = crest_elevation* |
+------------------+---------------------------------------------------------------------+

The values stored for the links are as follows:

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


[grass]
-------

Setting those parameters allows to run simulation outside the GRASS shell.
This is especially useful for batch processing involving different locations and mapsets.
If Itzï is run from within the GRASS shell, this section is not necessary.

+--------------+---------------------------------------------+---------+
| Keyword      | Description                                 | Format  |
+==============+=============================================+=========+
| grass\_bin   | Path to the grass binary                    | string  |
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

With GNU/Linux, *grass\_bin* could simply be *grass*.

The *region* and *mask* parameters are optionals and are applied only during the simulation.
After the simulation, those parameters are returned to the previous *region* and *mask* setting.
