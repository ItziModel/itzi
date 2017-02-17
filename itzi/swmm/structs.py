#! /usr/bin/python
# coding=utf8

import ctypes as c

class NodeData(c.Structure):
    '''Node results'''
    _fields_ = [
        ('inflow', c.c_double),       # total inflow (cfs)
        ('outflow', c.c_double),      # total outflow (cfs)
        ('head', c.c_double),         # Hydraulic head (invertElev + newDepth)
        ('crestElev', c.c_double),    # max elevation of the node (invertElev + fullDepth)
        ('type', c.c_int),            # node type code
        ('subIndex', c.c_int),        # index of node sub-category
        ('invertElev', c.c_double),   # invert elevation (ft)
        ('initDepth', c.c_double),    # initial storage level (ft)
        ('fullDepth', c.c_double),    # dist. from invert to surface (ft)
        ('surDepth', c.c_double),     # added depth under surcharge (ft)
        ('pondedArea', c.c_double),   # area filled by ponded water (ft2)
        ('degree', c.c_int),          # number of outflow links
        ('updated', c.c_char),        # true if state has been updated
        ('crownElev', c.c_double),    # top of highest connecting conduit (ft)
        ('losses', c.c_double),       # evap + exfiltration loss (ft3)
        ('newVolume', c.c_double),    # current volume (ft3)
        ('fullVolume', c.c_double),   # max. storage available (ft3)
        ('overflow', c.c_double),     # overflow rate (cfs)
        ('newDepth', c.c_double),     # current water depth (ft)
        ('newLatFlow', c.c_double)]   # current lateral inflow (cfs)


class LinkData(c.Structure):
    '''Link results'''
    _fields_ = [('flow', c.c_double),
                ('depth', c.c_double),
                ('velocity', c.c_double),
                ('volume', c.c_double),
                #~ ('shearVelocity', c.c_double),
                ('type', c.c_int),        # link type code
                #~ ('node1', c.c_char),      # start node ID
                #~ ('node2', c.c_char),      # end node ID
                ('offset1', c.c_double),  # ht. above start node invert (ft)
                ('offset2', c.c_double),  # ht. above end node invert (ft)
                ('yFull', c.c_double),    # depth when full (ft)
                ('froude', c.c_double)]   # Froude number


class ObjectType:
    GAGE = 0         # rain gage
    SUBCATCH = 1     # subcatchment
    NODE = 2         # conveyance system node
    LINK = 3         # conveyance system link
    POLLUT = 4       # pollutant
    LANDUSE = 5      # land use category
    TIMEPATTERN = 6  # dry weather flow time pattern
    CURVE = 7        # generic table of values
    TSERIES = 8      # generic time series of values
    CONTROL = 9      # conveyance system control rules
    TRANSECT = 10    # irregular channel cross-section
    AQUIFER = 11     # groundwater aquifer
    UNITHYD = 12     # RDII unit hydrograph
    SNOWMELT = 13    # snowmelt parameter set
    SHAPE = 14       # custom conduit shape
    LID = 15         # LID treatment units


class NodeType:
    JUNCTION = 0
    OUTFALL = 1
    STORAGE = 2
    DIVIDER = 3


class LinkType:
    CONDUIT = 0
    PUMP = 1
    ORIFICE = 2
    WEIR = 3
    OUTLET = 4
