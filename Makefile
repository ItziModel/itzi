MODULE_TOPDIR = ../..

PGM = t.itzi.sim

ETCFILES = simulation gis itzi_error \
           domain infiltration

#~ LIBES = /usr/include/python2.7/Python.h
DEPENDENCIES = $(GRASS_PYTHONPATH)

include $(MODULE_TOPDIR)/include/Make/Module.make
include $(MODULE_TOPDIR)/include/Make/Script.make
include $(MODULE_TOPDIR)/include/Make/Python.make

default: cmd
#~ default: script
