
====
Itzï
====

Itzï is a dynamic, fully distributed hydrologic and hydraulic model that
simulates 2D surface flows on a regular raster grid and drainage flows through the SWMM model.
It uses GRASS GIS as a back-end for entry data and result writing.

Website: http://www.itzi.org/

Documentation: http://itzi.rtfd.io/

Repository: https://bitbucket.org/itzi-model/itzi/


Description
===========

Itzï allows to simulate surface flows from direct rainfall or user-given point inflows.
It GIS integration simplifies the pre- and post-processing of data.
Notably, to change the resolution or the extend of the computational domain,
the user just need to change the GRASS computational region and the raster mask, if applicable.
This means that map export or re-sampling is not needed.

Itzï uses raster time-series as entry data, allowing, for example, the use of radar rainfall.


Model description
=================

Itzï is described in details in the following open-access article:

    Courty, L. G., Pedrozo-Acuña, A., & Bates, P. D. (2017).
    Itzï (version 17.1): an open-source, distributed GIS model for dynamic flood simulation.
    Geoscientific Model Development, 10(4), 1835–1847.
    http://doi.org/10.5194/gmd-10-1835-2017


Usage
=====

Simulations are set through a parameter file.

Simulation time
---------------

Simulation duration could be given by a combination of start time, end time and duration.
If only the duration is given, the results will be written as relative time STRDS.
In case start time is given, the simulation will use a absolute temporal type.
*record_step* is the time-step at which results are written to the disk.

Input data
----------

Itzï does not support Lat-Long coordinates.
A projected location should be used.
The inputs maps could be given either as STRDS or single maps.
First, the module try to load a STRDS of the given name.
If unsuccessful, it will load the given map, and stop with an error if the name does not correspond to either a map or a STRDS.

The following raster maps are necessary:

  * Digital elevation model in meters
  * Friction, expressed as Manning's n

The other space-time raster datasets are optionals.

Output data
-----------

A space-time raster dataset is created for each selected output.


============
Installation
============

Itzï depends on GRASS GIS 7 and NumPy.
GRASS should therefore be installed in order to use Itzï.
GRASS normally depends on NumPy, so installing manually NumPy might not be necessary.
All other dependencies are installed by pip.

Download and install Itzï::

    $ pip install itzi

Check usage of itzi command line::

    $ itzi -h

Run a simulation::

    $ itzi run parameters_file.ini

