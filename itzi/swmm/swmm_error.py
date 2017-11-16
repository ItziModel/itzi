#   coding=utf8

class SwmmError(Exception):
    '''Report swmm error message by taking the error code as input.
    Error codes and messages taken from error.c
    '''

    ERR101 = "ERROR 101: memory allocation error."
    ERR103 = "ERROR 103: cannot solve KW equations for Link %s."
    ERR105 = "ERROR 105: cannot open ODE solver."
    ERR107 = "ERROR 107: cannot compute a valid time step."

    ERR108 = "ERROR 108: ambiguous outlet ID name for Subcatchment %s."
    ERR109 = "ERROR 109: invalid parameter values for Aquifer %s."
    ERR110 = "ERROR 110: ground elevation is below water table for Subcatchment %s."

    ERR111 = "ERROR 111: invalid length for Conduit %s."
    ERR112 = "ERROR 112: elevation drop exceeds length for Conduit %s."
    ERR113 = "ERROR 113: invalid roughness for Conduit %s."
    ERR114 = "ERROR 114: invalid number of barrels for Conduit %s."
    ERR115 = "ERROR 115: adverse slope for Conduit %s."
    ERR117 = "ERROR 117: no cross section defined for Link %s."
    ERR119 = "ERROR 119: invalid cross section for Link %s."
    ERR121 = "ERROR 121: missing or invalid pump curve assigned to Pump %s."
    ERR122 = "ERROR 122: startup depth not higher than shutoff depth for Pump %s."

    ERR131 = "ERROR 131: the following links form cyclic loops in the drainage system:"
    ERR133 = "ERROR 133: Node %s has more than one outlet link."
    ERR134 = "ERROR 134: Node %s has illegal DUMMY link connections."

    ERR135 = "ERROR 135: Divider %s does not have two outlet links."
    ERR136 = "ERROR 136: Divider %s has invalid diversion link."
    ERR137 = "ERROR 137: Weir Divider %s has invalid parameters."
    ERR138 = "ERROR 138: Node %s has initial depth greater than maximum depth."
    ERR139 = "ERROR 139: Regulator %s is the outlet of a non-storage node."
    ERR141 = "ERROR 141: Outfall %s has more than 1 inlet link or an outlet link."
    ERR143 = "ERROR 143: Regulator %s has invalid cross-section shape."
    ERR145 = "ERROR 145: Drainage system has no acceptable outlet nodes."

    ERR151 = "ERROR 151: a Unit Hydrograph in set %s has invalid time base."
    ERR153 = "ERROR 153: a Unit Hydrograph in set %s has invalid response ratios."
    ERR155 = "ERROR 155: invalid sewer area for RDII at node %s."

    ERR156 = "ERROR 156: ambiguous station ID for Rain Gage %s."
    ERR157 = "ERROR 157: inconsistent rainfall format for Rain Gage %s."
    ERR158 = "ERROR 158: time series for Rain Gage %s is also used by another object."
    ERR159 = "ERROR 159: recording interval greater than time series interval for Rain Gage %s."

    ERR161 = "ERROR 161: cyclic dependency in treatment functions at node %s."

    ERR171 = "ERROR 171: Curve %s has invalid or out of sequence data."
    ERR173 = "ERROR 173: Time Series %s has its data out of sequence."

    ERR181 = "ERROR 181: invalid Snow Melt Climatology parameters."
    ERR182 = "ERROR 182: invalid parameters for Snow Pack %s."

    ERR183 = "ERROR 183: no type specified for LID %s."
    ERR184 = "ERROR 184: missing layer for LID %s."
    ERR185 = "ERROR 185: invalid parameter value for LID %s."
    ERR186 = "ERROR 186: invalid parameter value for LID placed in Subcatchment %s."
    ERR187 = "ERROR 187: LID area exceeds total area for Subcatchment %s."
    ERR188 = "ERROR 188: LID capture area exceeds total impervious area for Subcatchment %s."

    ERR191 = "ERROR 191: simulation start date comes after ending date."
    ERR193 = "ERROR 193: report start date comes after ending date."
    ERR195 = "ERROR 195: reporting time step or duration is less than routing time step."

    ERR200 = "ERROR 200: one or more errors in input file."
    ERR201 = "ERROR 201: too many characters in input line "
    ERR203 = "ERROR 203: too few items "
    ERR205 = "ERROR 205: invalid keyword %s "
    ERR207 = "ERROR 207: duplicate ID name %s "
    ERR209 = "ERROR 209: undefined object %s "
    ERR211 = "ERROR 211: invalid number %s "
    ERR213 = "ERROR 213: invalid date/time %s "
    ERR217 = "ERROR 217: control rule clause invalid or out of sequence "  # (5.1.008)
    ERR219 = "ERROR 219: data provided for unidentified transect "
    ERR221 = "ERROR 221: transect station out of sequence "
    ERR223 = "ERROR 223: Transect %s has too few stations."
    ERR225 = "ERROR 225: Transect %s has too many stations."
    ERR227 = "ERROR 227: Transect %s has no Manning's N."
    ERR229 = "ERROR 229: Transect %s has invalid overbank locations."
    ERR231 = "ERROR 231: Transect %s has no depth."
    ERR233 = "ERROR 233: invalid treatment function expression "

    ERR301 = "ERROR 301: files share same names."
    ERR303 = "ERROR 303: cannot open input file."
    ERR305 = "ERROR 305: cannot open report file."
    ERR307 = "ERROR 307: cannot open binary results file."
    ERR309 = "ERROR 309: error writing to binary results file."
    ERR311 = "ERROR 311: error reading from binary results file."

    ERR313 = "ERROR 313: cannot open scratch rainfall interface file."
    ERR315 = "ERROR 315: cannot open rainfall interface file %s."
    ERR317 = "ERROR 317: cannot open rainfall data file %s."
    ERR318 = "ERROR 318: date out of sequence in rainfall data file %s."
    ERR319 = "ERROR 319: unknown format for rainfall data file %s."
    ERR320 = "ERROR 320: invalid format for rainfall interface file."
    ERR321 = "ERROR 321: no data in rainfall interface file for gage %s."

    ERR323 = "ERROR 323: cannot open runoff interface file %s."
    ERR325 = "ERROR 325: incompatible data found in runoff interface file."
    ERR327 = "ERROR 327: attempting to read beyond end of runoff interface file."
    ERR329 = "ERROR 329: error in reading from runoff interface file."

    ERR330 = "ERROR 330: hotstart interface files have same names."
    ERR331 = "ERROR 331: cannot open hotstart interface file %s."
    ERR333 = "ERROR 333: incompatible data found in hotstart interface file."
    ERR335 = "ERROR 335: error in reading from hotstart interface file."

    ERR336 = "ERROR 336: no climate file specified for evaporation and/or wind speed."
    ERR337 = "ERROR 337: cannot open climate file %s."
    ERR338 = "ERROR 338: error in reading from climate file %s."
    ERR339 = "ERROR 339: attempt to read beyond end of climate file %s."

    ERR341 = "ERROR 341: cannot open scratch RDII interface file."
    ERR343 = "ERROR 343: cannot open RDII interface file %s."
    ERR345 = "ERROR 345: invalid format for RDII interface file."

    ERR351 = "ERROR 351: cannot open routing interface file %s."
    ERR353 = "ERROR 353: invalid format for routing interface file %s."
    ERR355 = "ERROR 355: mis-matched names in routing interface file %s."
    ERR357 = "ERROR 357: inflows and outflows interface files have same name."

    ERR361 = "ERROR 361: could not open external file used for Time Series %s."
    ERR363 = "ERROR 363: invalid data in external file used for Time Series %s."

    ERR401 = "ERROR 401: general system error."
    ERR402 = "ERROR 402: cannot open new project while current project still open."
    ERR403 = "ERROR 403: project not open or last run not ended."
    ERR405 = ("ERROR 405: amount of output produced will exceed maximum file size; "
              "either reduce Ending Date or increase Reporting Time Step.")

    ErrorMsgs = ["", ERR101, ERR103, ERR105, ERR107, ERR108, ERR109, ERR110, ERR111,
                 ERR112, ERR113, ERR114, ERR115, ERR117, ERR119, ERR121, ERR122, ERR131,
                 ERR133, ERR134, ERR135, ERR136, ERR137, ERR138, ERR139, ERR141, ERR143,
                 ERR145, ERR151, ERR153, ERR155, ERR156, ERR157, ERR158, ERR159, ERR161,
                 ERR171, ERR173, ERR181, ERR182, ERR183, ERR184, ERR185, ERR186, ERR187,
                 ERR188, ERR191, ERR193, ERR195, ERR200, ERR201, ERR203, ERR205, ERR207,
                 ERR209, ERR211, ERR213, ERR217, ERR219, ERR221, ERR223, ERR225, ERR227,
                 ERR229, ERR231, ERR233, ERR301, ERR303, ERR305, ERR307, ERR309, ERR311,
                 ERR313, ERR315, ERR317, ERR318, ERR319, ERR320, ERR321, ERR323, ERR325,
                 ERR327, ERR329, ERR330, ERR331, ERR333, ERR335, ERR336, ERR337, ERR338,
                 ERR339, ERR341, ERR343, ERR345, ERR351, ERR353, ERR355, ERR357, ERR361,
                 ERR363, ERR401, ERR402, ERR403, ERR405]

    def __init__(self, errcode):
        self.msg = SwmmError.ErrorMsgs[errcode]
    def __str__(self):
        return repr(self.msg)


class NotOpenError(Exception):
    def __str__(self):
        return repr('SWMM file should be open')


class NotStartedError(Exception):
    def __str__(self):
        return repr('SWMM simulation must be started')


class LinkTypeError(Exception):
    """Error related to linkage type
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)
