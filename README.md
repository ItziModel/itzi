# t.sim.flood
A GRASS GIS 7 module that simulates 2D superficial flows using simplified shallow water equations

# Description
This module is aimed at modelling floodplain inundations using simplified shallow water equations.
It implements the q-centered numerical scheme described in:

De Almeida, G. a M. et al., 2012.
Improving the stability of a simple formulation of the shallow water equations for 2-D flood modeling.
Water Resources Research, 48(5), pp.1–14.

De Almeida, G. a M. & Bates, P., 2013.
Applicability of the local inertial approximation of the shallow water equations to flood modeling.
Water Resources Research, 49(8), pp.4833–4844.

As well as the simple routing method explained in:
Sampson, C.C. et al., 2013.
An automated routing methodology to enable direct rainfall in high resolution shallow water models.
Hydrological Processes, 27(3), pp.467–476.

It outputs space-time raster datasets of:
  - water depth
  - water surface elevation (depth + DEM)

# Validity
To be completed

# Usage
## Input datas
The following raster maps are necessary:
  - Digital elevation model in meters
  - Friction, expressed as Manning's n

The following raster space-time datasets are optional:
  - rain map in mm/h
  - user defined inflow in m/s (vertical velocity, i.e for 20m3/s on a 10x10 cell, the velocity is 0.2 m/s)

Note: for now, only relative time in days, hours, minutes or seconds is accepted

The following raster maps are optionals:
  - boundary condition type: an integer map (see below)
  - boudary condition value: a map for boundary conditions using user-given value

The following informations are needed to start the simulation:
  - sim_duration: the total simulated time in seconds
  - record_step: the step in which results maps are written to the drive

## Boundary values
  Only the cells on the edge of the map are used by the software. All other values are ignored
  The boudary type is defined by the following cell values:
  - 1: closed: flow = 0
  - 2: open: z=neighbour, depth=neighbour, v=neighbour
  - 3: fixed_h: z=neighbour,  water surface elevation=user defined, q=variable
  
  The boundary value map is used in the case of fixed value boundary condition

## Output datas
The user can choose to output the following:
  - water depth
  - water surface elevation (depth + DEM)

A raster space-time dataset is created for each selected output.
Maps are written using the STRDS name as a prefix.

# Installation
t.sim.flood depends on GRASS GIS 7 and Cython. Numpy is needed but is normally a GRASS dependency.
Copy all the source in the directory of your choice.

Run cython:
$ cython hydro_cython.pyx

Compile the generated C file
$ gcc -shared -pthread -fPIC -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o hydro_cython.so hydro_cython.c

Launch GRASS
$ grass70

Check usage of the module
$ python t.sim.flood.py --h

# Known issues

  - Fixed-water surface elevation boundary is not working properly on East and North region border
  - NULL cells are not handled. Any map containing such cell would lead to the program generating unpredictable results
  - Instabilities and negatives depths values may happen in high slopes regions.
