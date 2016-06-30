
====
Itzï
====

Itzï is a dynamic, fully distributed hydrologic and hydraulic model that
simulates 2D superficial flows on a regular raster grid using simplified shallow water equations.
It uses GRASS GIS as a back-end for entry data and result writing.

Description
===========

Itzï allows to simulate superficial flows from direct rainfall or user-given point inflows.
It GIS integration simplifies the pre- and post-processing of data.
Notably, to change the resolution or the extend of the computational domain,
the user just need to change the GRASS computational region and the raster mask, if applicable.
This means that map export or re-sampling is not needed.

Itzï uses raster time-series as entry data, allowing the uses of radar rainfall or varying friction coefficient.

In Itzï, the simulation time information could be represented in two ways.
First, the time could relative, i.e given as a number of hours, minutes and seconds, whatever the actual date is.
Alternatively, the user can provide a start date and time, together with a duration or a end time.
This second way is useful to simulate historical events with a known date,
especially with input data with such time representation, for example rainfall.
To be able to use this functionality, the time-series given as entry data should use the same temporal reference system.

Superficial flow model
======================

Itzï implements a partial inertia, finite-difference numerical scheme described in:

De Almeida, G. a M. et al., 2012.
Improving the stability of a simple formulation of the shallow water equations for 2-D flood modeling.
Water Resources Research, 48(5), pp.1–14.

De Almeida, G. a M. & Bates, P., 2013.
Applicability of the local inertial approximation of the shallow water equations to flood modeling.
Water Resources Research, 49(8), pp.4833–4844.

As well as a simple rain routing method inspired by:

Sampson, C.C. et al., 2013.
An automated routing methodology to enable direct rainfall in high resolution shallow water models.
Hydrological Processes, 27(3), pp.467–476.


Infiltration model
==================

Itzï can simulate the infiltration on two different ways:
    * Green-Ampt infiltration model.
    * User given infiltration rate in mm/h


Usage
=====

Simulations parameters are given through a file written in a style
similar to the Microsoft Windows INI files.

Simulation time
---------------

Simulation duration could be given by a combination of start time, end time and duration.
If only the duration is given, the results will be written as relative time STRDS.
In case start time is given, the simulation will use a absolute temporal type.
*record_step* is the time-step at which results are written to the disk.

Input data
----------

Itzï does not support Lat-Long coordinates. A projected location should be used.
The inputs maps could be given either as STRDS or single maps.
First, the module try to load a STRDS of the given name.
If unsuccessful, it will load the given map, and stop with an error if the name does not correspond to either a map or a STRDS.

The following raster maps are necessary:

  * Digital elevation model in meters
  * Friction, expressed as Manning's n

The following space-time raster datasets are optional:

  * rain map in mm/h
  * point inflow in m/s (vertical velocity, i.e for 20 m3/s on a 10x10 cell, the velocity is 0.2 m/s)
  * fixed infiltration rate in mm/h
  * Green-Ampt infiltration parameters
  * boundary condition type: an integer map (see below)
  * boundary condition value: a map for boundary conditions using user-given value


Boundary conditions
-------------------

  The boundary type is defined by the following cell values:

  1. closed: flow = 0
  2. open: velocity at the boundary is equal to the velocity inside the domain
  3. fixed-depth: not implemented yet
  4. user-defined water depth inside the domain

  The boundary value map is used in the case of types 3 and 4 boundary condition.
  The open boundary condition is experimental and not well tested.

Output data
-----------

The user can choose to output the following:

  * water depth
  * water surface elevation (depth + DEM)
  * velocity magnitude in m/s and direction in degrees
  * x and y flows in m3/s
  * statistical maps of boundaries, infiltration, rainfall and/or point inflow

A raster space-time dataset is created for each selected output.
Maps are written using the STRDS name as a prefix.
Flows values are the values at E and S boundaries of the given cell, respectively.
Values of statistical maps corresponds to the average value during the last record interval.


============
Installation
============

Itzï depends on GRASS GIS 7.
GRASS should therefore be installed in order to use Itzï.
All other dependencies are installed by pip.

Install Itzï::

    $ pip install itzi

Launch GRASS::

    $ grass

Check usage of itzi command line::

    $ itzi -h

Run a simulation::

    $ itzi run parameters_file.ini

