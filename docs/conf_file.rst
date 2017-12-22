
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
| friction                | Manning's *n* (friction coefficient)    | map or strds |
+-------------------------+-----------------------------------------+--------------+
| start\_h                | Starting water depth in meters          | map name     |
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
| effective\_pororosity   | Effective porosity in mm/mm             | map or strds |
+-------------------------+-----------------------------------------+--------------+
| capillary\_pressure     | Wetting front capillary pressure head   | map or strds |
|                         | in mm                                   |              |
+-------------------------+-----------------------------------------+--------------+
| hydraulic\_conductivity | Soil hydraulic conductivity in mm/h     | map or strds |
+-------------------------+-----------------------------------------+--------------+
| losses                  | User-defined losses in mm/h             | map or strds |
|                         | (*new in16.9, renamed in 17.7*)         |              |
+-------------------------+-----------------------------------------+--------------+

.. deprecated:: 17.7
    *drainage\_capacity* is renamed to *losses*

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

+--------------+---------------------------------------------------------+--------+
| Keyword      | Description                                             | Format |
+==============+=========================================================+========+
| h            | Water depth                                             | meters |
+--------------+---------------------------------------------------------+--------+
| wse          | Water surface elevation (depth + elevation)             | meters |
+--------------+---------------------------------------------------------+--------+
| v            | Overland flow speed (velocity's magnitude)              | m/s    |
+--------------+---------------------------------------------------------+--------+
| vdir         | Velocity's direction. CCW from East                     | degrees|
|              |                                                         |        |
+--------------+---------------------------------------------------------+--------+
| qx           | Volumetric flow, x direction. Positive if going East    | m³/s   |
+--------------+---------------------------------------------------------+--------+
| qy           | Volumetric flow, y direction. Positive if going South   | m³/s   |
+--------------+---------------------------------------------------------+--------+
| boundaries   | Flow coming in (positive) or going out (negative) the   | m/s    |
|              | domain due to boundary conditions. Average since the    |        |
|              | last record                                             |        |
+--------------+---------------------------------------------------------+--------+
| infiltration | Infiltration rate. Average since the last record        | mm/h   |
|              |                                                         |        |
+--------------+---------------------------------------------------------+--------+
| rainfall     | Rainfall rate. Average since the last record            | mm/h   |
+--------------+---------------------------------------------------------+--------+
| inflow       | Average user flow since the last record                 | m/s    |
+--------------+---------------------------------------------------------+--------+
| losses       | Average losses since the last record                    | m/s    |
|              | (*new in 17.1, renamed in 17.7*)                        |        |
+--------------+---------------------------------------------------------+--------+
|drainage_stats| Average exchange flow between surface and drainage model|        |
|              | since the last record (*new in 17.7*)                   | m/s    |
+--------------+---------------------------------------------------------+--------+
| verror       | Total created volume due to numerical error since the   | m³     |
|              | last record (*new in 17.1*)                             |        |
+--------------+---------------------------------------------------------+--------+

.. versionadded:: 17.1
    *drainage_cap* and *verror* are added.

.. versionchanged:: 17.7
    *drainage_cap* is renamed to *losses*

Additionally to output a map at each *record\_step*, *h* and *v* also
produce a map of maximum values.

.. note:: Water depth maps, apart from map of maximum values,
    do not display values under the *hmin* threshold (See below).
    When the exported map is totally empty, it is deleted at the end of the simulation.

[statistics]
------------

+---------------+-------------------+-------------+
| Keyword       | Description       | Format      |
+===============+===================+=============+
| stats\_file   | Statistics file   | CSV table   |
+---------------+-------------------+-------------+

Statistics file
^^^^^^^^^^^^^^^
.. versionchanged:: 17.1
    Mass balance calculation now takes into account the volume from losses.
    Created volume calculation is changed.

The statistic file is presented as a CSV file and updated at each *record_step*.
The values exported are shown in the table below.

Water entering the domain is represented by a positive value.
Water that leaves the domain is negative.
Volumes are in m³.

+-----------------+------------------------------------------------------------------+
| Keyword         | Description                                                      |
+=================+==================================================================+
| sim\_time       | Elapsed simulation time                                          |
+-----------------+------------------------------------------------------------------+
| avg\_timestep   | Average time-step duration since last record                     |
+-----------------+------------------------------------------------------------------+
| #timesteps      | Number of time-steps since the last record                       |
+-----------------+------------------------------------------------------------------+
| boundary\_vol   | Water volume that passed the domain boundaries since last record |
+-----------------+------------------------------------------------------------------+
| rain\_vol       | Rain volume that entered the domain since last record            |
+-----------------+------------------------------------------------------------------+
| inf\_vol        | Water volume that left the domain due to infiltration since      |
|                 | last record                                                      |
+-----------------+------------------------------------------------------------------+
| inflow\_vol     | Water volume that entered or left the domain due to user         |
|                 | inflow since last record                                         |
+-----------------+------------------------------------------------------------------+
| losses\_vol     | Water volume that entered or left the domain due to              |
|                 | losses since last record                                         |
+-----------------+------------------------------------------------------------------+
| drain\_net\_vol | Water volume that entered or left the surface domain since       |
|                 | last record due to exchanges with the drainage network           |
+-----------------+------------------------------------------------------------------+
| domain\_vol     | Total water volume in the domain at this time-step               |
+-----------------+------------------------------------------------------------------+
| created\_vol    | Water volume created due to numerical errors since last record   |
|                 | record                                                           |
+-----------------+------------------------------------------------------------------+
| %error          | Percentage of the domain volume variation due to numerical       |
|                 | error. Corresponds to *created\_vol* / (*domain\_vol* -          |
|                 | *old\_domain\_vol*) \* 100                                       |
+-----------------+------------------------------------------------------------------+

.. versionchanged:: 17.7
    *drain_cap_vol* is renamed to *losses_vol*

.. versionadded:: 17.7
    *drain_net_vol* is added.


[options]
---------

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

When water depth is under *hmin*, the flow is routed at the fixed velocity defined by *vrouting*.


[drainage]
----------

.. versionadded:: 17.7

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

.. versionadded:: 17.11
    *orifice_coeff*, *free_weir_coeff* and *submerged_weir_coeff* are added.

The output maps are organised in two layers.
The nodes are stored in layer 1, the links in layer 2.

The values stored for the nodes are described below. All are instantaneous.

+--------------+---------------------------------------------------------+
| Column       | Description                                             |
+==============+=========================================================+
| cat          | DB key                                                  |
+--------------+---------------------------------------------------------+
| node_id      | Name of the node                                        |
+--------------+---------------------------------------------------------+
| type         | Node type  (junction, storage, outlet etc.)             |
+--------------+---------------------------------------------------------+
| linkage_type | Equation used for the drainage/surface linkage          |
+--------------+---------------------------------------------------------+
| linkage_flow | Flow moving from the drainage to the surface            |
+--------------+---------------------------------------------------------+
| inflow       | Flow entering the node (m³/s)                           |
+--------------+---------------------------------------------------------+
| outflow      | Flow exiting the node (m³/s)                            |
+--------------+---------------------------------------------------------+
| latFlow      | SWMM lateral flow (m³/s)                                |
+--------------+---------------------------------------------------------+
| head         | Hydraulic head in metre                                 |
+--------------+---------------------------------------------------------+
| crownElev    | Elevation of the highest crown of the connected conduits|
+--------------+---------------------------------------------------------+
| crestElev    | Elevation of the top of the node in metres              |
+--------------+---------------------------------------------------------+
| invertElev   | Elevation of the bottom of the node in metres           |
+--------------+---------------------------------------------------------+
| initDepth    | Water depth in the node at the start of the simulation  |
+--------------+---------------------------------------------------------+
| fullDepth    | *crownElev* - *invertElev* (m)                          |
+--------------+---------------------------------------------------------+
| surDepth     | Depth above *crownElev* before overflow begins          |
+--------------+---------------------------------------------------------+
| pondedArea   | Area above the node where ponding occurs (m²)           |
+--------------+---------------------------------------------------------+
| degree       | Number of pipes connected to the node                   |
+--------------+---------------------------------------------------------+
| newVolume    | Water volume in the node                                |
+--------------+---------------------------------------------------------+
| fullVolume   | Volume in the node when *head - invertElev = crestElev* |
+--------------+---------------------------------------------------------+

The values stored for the links are as follows:

+--------------+-------------------------------------------------------+
| Column       | Description                                           |
+==============+=======================================================+
| cat          | DB key                                                |
+--------------+-------------------------------------------------------+
| link_id      | Name of the link                                      |
+--------------+-------------------------------------------------------+
| type         | Link type (conduit, pump etc.)                        |
+--------------+-------------------------------------------------------+
| flow         | Volumetric flow (m³/s)                                |
+--------------+-------------------------------------------------------+
| depth        | Water depth in the conduit (m)                        |
+--------------+-------------------------------------------------------+
| velocity     | Average flow velocity (m/s)                           |
+--------------+-------------------------------------------------------+
| volume       | Water volume stored in the conduit (m³)               |
+--------------+-------------------------------------------------------+
| offset1      | Height above inlet node invert elevation (m)          |
+--------------+-------------------------------------------------------+
| offset2      | Height above outlet node invert elevation (m)         |
+--------------+-------------------------------------------------------+
| yFull        | Average water depth when the pipe is full (m)         |
+--------------+-------------------------------------------------------+
| froude       | Average Froude number                                 |
+--------------+-------------------------------------------------------+


[grass]
-------

.. versionadded:: 16.9

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

.. versionadded:: 17.11
    *region* and *mask* are added.

With GNU/Linux, *grass\_bin* could be simply *grass*.

The *region* and *mask* parameters are optionals and are applied only during the simulation.
After the simulation, those parameters are returned to the previous *region* and *mask* setting.
