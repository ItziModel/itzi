
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

Itzï allows the simulation of surface flows from direct rainfall or user-given point inflows.
Its GIS integration simplifies the pre- and post-processing of data.
Notably, to change the resolution or the extend of the computational domain,
the user just need to change the GRASS computational region and the raster mask, if applicable.
This means that map export or re-sampling is not needed.

Itzï uses raster time-series as entry data, allowing the use of rainfall from weather radars.


Model description
=================

The surface model of Itzï is described in details in the following open-access article:

    Courty, L. G., Pedrozo-Acuña, A., & Bates, P. D. (2017).
    Itzï (version 17.1): an open-source, distributed GIS model for dynamic flood simulation.
    Geoscientific Model Development, 10(4), 1835–1847.
    http://doi.org/10.5194/gmd-10-1835-2017

