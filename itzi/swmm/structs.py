# coding=utf8

class ObjectType(object):
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


class NodeType(object):
    JUNCTION = 0
    OUTFALL = 1
    STORAGE = 2
    DIVIDER = 3


class LinkType(object):
    CONDUIT = 0
    PUMP = 1
    ORIFICE = 2
    WEIR = 3
    OUTLET = 4


# correspondence between C enum and python string
LINK_TYPES = {LinkType.CONDUIT: "conduit",
              LinkType.PUMP: "pump",
              LinkType.ORIFICE: "orifice",
              LinkType.WEIR: "weir",
              LinkType.OUTLET: "outlet"}

NODE_TYPES = {NodeType.STORAGE: 'storage',
              NodeType.JUNCTION: 'junction',
              NodeType.OUTFALL: 'outfall',
              NodeType.DIVIDER: 'divider'}

ROUTING_MODELS = {0: 'NO_ROUTING',  # no routing
                  1: 'SF',  # steady flow model
                  2: 'KW',  # kinematic wave model
                  3: 'EKW',  # extended kin. wave model
                  4: 'DW'}  # Dynamic wave

LINKAGE_TYPES = {0: "not linked",
                 1: 'linked, no flow',
                 2: 'free weir',
                 3: 'submerged weir',
                 4: 'orifice'}
