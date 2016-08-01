
=============
Release Notes
=============

These are the major changes made in each release.
For details of the changes please see the commit log of the git repository.

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
