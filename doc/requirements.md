# Requirements for t.sim.flood

GRASS GIS module aiming at simulate overland flow using a finite-differences
on staggered grid numerical scheme, as described by:

De Almeida, G. a M. et al., 2012. 
Improving the stability of a simple formulation of the shallow water
equations for 2-D flood modeling. Water Resources Research, 48(5), pp.1–14.

De Almeida, G. a M. & Bates, P., 2013. 
Applicability of the local inertial approximation of the shallow water
equations to flood modeling. Water Resources Research, 49(8), pp.4833–4844.

The main goal of t.sim.flood is to be integrated with SWMM drainage model.

## Inputs

- raster maps
    - DEM in m
    - Manning's n friction
    - Start depth at the beginning of the simulation
    - Boundary condition type
- space-time raster datasets
    - Boundary condition values, unit depending on the type
    - Rainfall in mm/h
    - User defined flow in m/s
- other
    - simulation start date and time
    - simulation duration / end time
    - record interval (relative time)
- runtime input:
    - input point flow value from SWMM simulation

Any type of map input could be provided as a STRDS or a single raster map

## Outputs

- space-time raster datasets (time granularity user-defined)
    - Water depth in m
    - water surface elevation in m (DEM + water depth)
    - Water velocities in x direction (m/s)
    - Water velocities in y direction (m/s)
-  runtime output
    - simulation progress (print to shell)
    - inflows for input of SWMM simulation

## Interfaces

### GRASS GIS

- Read and write GRASS rasters and space-time datasets
- Be integrated as a module
- Use GRASS module standard user-interface
- Use GRASS messenger for user outputs
- Use GRASS coding and testing practices
- Use GRASS compatible licence

### SWMM

Be able to receive and export values at runtime with SWMM
__to be completed: which exchange format?__

## Definition of success

- Results comparable with LSFLOOD-FP and real-world values
- Run-time at least comparable with LISFLOOD for similar simulation
- Smooth coupling with SWMM possible
