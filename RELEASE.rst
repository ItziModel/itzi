
=============
Release Notes
=============

These are the major changes made in each release.
For details of the changes please see the commit log of the git repository.

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

- Allow easy installation and compilation with pip

**New user interface**

- Parameters are now given only by configuration file
- Parameters name in configuration files are more explicit
- Output maps are now defined by a prefix and a list of output needed
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
