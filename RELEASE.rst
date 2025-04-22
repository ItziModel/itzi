
=============
Release Notes
=============

These are the major changes made in each release.
For more details please see the commit log of the git repository.


Itzï 25.4
---------

*Release date: 2025-04-21*

This is a modernized version of itzi.
It implements the `Basic Model Interface <https://csdms.colorado.edu/wiki/BMI>`__,
allowing the coupling of itzi with other models.
This version requires GRASS 8.4 or above.

**Improvements**

- Add compatibility with the `Basic Model Interface <https://csdms.colorado.edu/wiki/BMI>`__.

**Corrected bugs**

- Fix the calculation of the Green-Ampt infiltration rate.

**Internals**

- This version now relies on pyswmm for the drainage coupling, instead of embedding the SWMM codebase.
- The code is now tested with Python 3.11 and above.
- Better code organization and refactoring.
- Remove the dependency on grass-session, as allowed by grass 8.4.
- More tests and better test coverage.
- Add a benchmark for the surface flow simulation.
- All units for infiltration are now m/s instead of mm/h, preventing conversion errors.
- More modern package tooling.


Itzï 20.5
---------

*Release date: 2020-05-26*

This version supports Python 3.7 and above, and therefore GRASS 7.8 and above.
Python 2.x is no longer supported.

**Improvements**

- Raster maps are now written in the background. This might improve the performance in some cases.

**Changes in the configuration file**

- The "effective_pororosity" option is renamed to "effective_porosity". Using the former will raise a deprecation warning.

**Corrected bugs**

- Fix the stencil for 2D friction. This do not seems to have significant real-word impact.
- Computation time display is now working correctly when running simulations with the same filename.
- The first map of input inflow series is now taken into account.
- Raster masks are now properly taken into account.

**Internals**

- Python 3.7 or above is now mandatory.
- Refactoring of the code to follow a initialize-run-finalize model.
- Inclusion of a test suite to improve quality insurance.
- Use of grass-session to set the GRASS environment.


Itzï 18.2
---------

*Release date: 2018-02-19*

This new version of Itzï improves the stability of the surface-drainage coupling.
The part of the code calculating the coupling has been mostly rewritten.
The coupling flow between the surface and the drainage is calculated as follows:

- In case of drainage overflow, the orifice equation is used
- In case of surface to drainage flow, the coupling flow is calculated by using either the orifice,
free weir or submerged weir equation.
The equation is chosen according to the relative water surface elevations in the surface and the drainage, and the node crest elevation.
- If the water is entering the drainage network, the flow is limited to prevent negative depth in the surface model.
- The coupling flow cannot invert in one time-step. It must spend one time-step at zero.
This reduces the oscillations that could occur when the water elevations in both the drainage network and the surface are similar.


Itzï 17.11
----------

*Release date: 2017-11-24*

**New features**

- Drainage coupling: Add the possibility to set the orifice and weir coefficients in the configuration file.
- Allow to set the raster mask and region from the input file.
- Now print an out-of-memory error message instead of a blank crash if the domain does not fit in the RAM.

**Corrected bugs**

- Drainage coupling: fix unit conversion problem in setting node fullDepth, set fullVolume at the same time.

**Code organization**

- Improve coding standard with pylint.
- Advance Python 3 support. Still limited by GRASS own incompatibility.
- Move more GRASS functions to gis.py.


Itzï 17.10
----------

*Release date: 2017-10-27*

**Corrected bugs**

- Fix problems of unit conversion in the interchange between the surface and the drainage model.
- Do not write the output vector maps if not wanted by the user.
- Fix a division by zero error appearing at the beginning of the simulation.


Itzï 17.8
---------

*Release date: 2017-08-08*

This is principally a bugfix release.

**New features**

- Flows interchanges when water is leaving the drainage network are always modelled with an orifice equation.
  This is in accordance with [1], from where the coefficients are taken.

**Corrected bugs**

- Fix the pip installation process. Remove the cython dependency.

[1] Rubinato et al. 2017. doi:10.1016/j.jhydrol.2017.06.024


Itzï 17.7
---------

*Release date: 2017-07-31*

This release adds the integration of the SWMM drainage network model.

**New features**

- Bi-directional coupling with the SWMM model
- Velocity is now calculated at the centre of the cells
- It is possible to output maps of the Froude number
- The maps of the initial state of the simulation are recorded

**Changes in the configuration file**

Some changes have been made to the configuration file in order to make the options clearer.

- In the [input] section, *drainage_capacity* is renamed *losses*
- In the [output] section, *drainage_cap* is renamed *losses*

If Itzï is run with an older option name, the user will receive a deprecation warning.
Those legacy options are set for deletion in a later release.
Please update your configuration files.

**Corrected bugs**

- Maps are recorded at the very end of the simulation
- Check if the domain is at least 3x3 cells before running a simulation.
- Itzï will check if grassdata, location and mapsets exist before running a simulation.


Itzï 17.1
---------

*Release date: 2017-01-31*

This is mainly a bugfix release

**Corrected bugs**

- Mass balance calculation now takes into account the volume from drainage capacity
- Volume error calculation is more accurate

**New features**

- Add the possibility to export the map of created volume from continuity error
- The *%error* column of the statistic output is now the percentage of the domain volume variation that is due to error


Itzï 16.9
---------

*Release date: 2016-10-03*

**New features**

- If multiple parameters files are given, they are run in a batch.
- Simulations can be run from outside GRASS.
- Add the possibility to set a drainage capacity map as entry data on top of infiltration parameters.
- The progress message is now more informative, giving the ETA and current simulation time.

**Installation**
- NumPy is no longer installed by default. This prevents pip from installing a new version of NumPy even if another is already installed.


Itzï 16.8
---------

*Release date: 2016-08-10*

This is mainly a bugfix release.

**Corrected bugs**

- fix crash when using absolute time
- fix crash when not providing a statistics file name
- clearer message in case mandatory parameters are not set

**New feature**

- Allow display of CLI usage outside of GRASS environment


Itzï 16.7
---------

*Release date: 2016-07-15*

This is the first release of Itzï on Pypi

**Easier installation**

- Easy compilation and installation with pip

**New user interface**

- Parameters are now given only by configuration file
- Parameters name in configuration files are more explicit
- Output maps are now defined by a prefix and a list of output
- Add an example input file with parameter description

**Corrected bugs**

- Exit nicely if not run within GRASS environment
- Return an error if the input parameter files is not found

**New features**

- Export statistical maps for boundary flows, user inflow, infiltration and rainfall rates

**Faster**

- More tasks are run in parallel
- Minimize memory access

**Known issues**

- Open boundary condition is experimental and only tested on the East domain boundary.
