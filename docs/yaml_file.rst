YAML configuration file
=======================

A ``.yaml`` file is a stream of ensemble documents.
One single ``.yaml`` file can contain multiple documents separated with ``---``.
Each non-empty document defines one ensemble.
Multiple YAML files and :doc:`INI configurations <conf_file>` can be passed to ``itzi run`` in
one command.
Other filename extensions are treated as INI.
YAML ensemble resume from hotstart is not currently supported.

The file must be encoded in UTF-8.
The supported subset rejects empty documents, duplicate
keys, merge keys, YAML directives, custom tags, non-JSON values, and unknown
options at every level.
Put quotes around values that should be text, such as timestamps, durations,
map names, paths, and output templates.
Without quotes, YAML may read values like ``yes`` as a boolean or ``2026-09-01`` as a date instead of text.
Leave numeric parameters unquoted (for example, ``cfl: 0.7``).
An invalid document does not prevent later explicitly delimited documents from being checked.

Example
-------

.. code-block:: yaml

   schema_version: 1
   ensemble:
     id: "central-city"
     description: "Central city rainfall study"
   grass:
     database: "/srv/grassdata"
     project: "central_city"
     mapset: "itzi"
     region: "central_city_5m"
   time:
     start: "2026-09-01T00:00:00"
     duration: "02:00:00"
     record_step: "00:05:00"
   input:
     ground_elevation: "elevation_5m@PERMANENT"
     friction: "manning_n@PERMANENT"
     rainfall_rate: ["rain_10yr@PERMANENT", "rain_100yr@PERMANENT"]
     infiltration:
       type: "none"
   parameters:
     cfl: [0.5, 0.7]
   outputs:
     rasters:
       prefix: "central_city_{simulation}"
       variables: [water_depth, flow_speed]
     statistics:
       file: "results/{simulation}.csv"

This produces four candidate simulations (two rainfall maps × two ``cfl`` values).
Use ``itzi run --dry-run study.yaml`` to expand and validate members
without writing a manifest or model outputs.

Document structure and sweeps
-----------------------------

The top-level sections are ``schema_version``, ``ensemble``, ``grass``,
``time``, ``input``, ``parameters``, ``drainage``, and ``outputs``. All except
``drainage`` are required.
Section names and option names are case-sensitive.

Each ``input`` map, ``parameters`` value, and ``drainage`` option can be a
scalar or a non-empty list of alternatives.
Lists are independent Cartesian dimensions, or sweep.
Duplicate values within a list are rejected.
``input.infiltration`` can instead be a non-empty list of whole model alternatives.
``grass``, ``time``, and output options are fixed across the ensemble.
``outputs.rasters.variables`` is a selection list.
At most 100 candidate simulations may come from one document and 200 from the whole command.
Ensemble IDs must be unique across a batch.

``schema_version``
------------------

Required integer, currently only ``1``. This is distinct from the
:doc:`manifest version <manifest>`.

``ensemble``
------------

* ``id`` (required): 1–80 characters, starting with an ASCII letter or digit;
  remaining characters may also be ``.``, ``_``, or ``-``. This identifies the
  ensemble and names its default manifest. It must be unique within a batch.
* ``description`` (optional): text up to 256 characters, included in the
  manifest.

``grass``
---------

Use ``grass: {}`` to use the active GRASS session, computational region, and mask.
An explicit session requires all three of the following options:

* ``database``: path to the GIS database.
* ``project``: GRASS project (location) name.
* ``mapset``: mapset name.

``executable`` optionally selects the GRASS executable for an explicit
session (defaults to ``grass``); it cannot be used without those three options.
``region`` optionally names a saved region; ``mask`` optionally names a raster mask.
Both can also be used to override the active region or mask for the run.
When a session is already active, an explicit database, project, and mapset must match it.
All members of an ensemble share this GRASS context and domain.
Only a projected location / project is accepted, not latitude/longitude.
The region must have at least 3 rows and 3 columns, and a mask cannot exclude the entire simulation domain.

``time``
--------

* ``record_step`` (required): interval between recorded outputs.
* ``duration``: length of the simulation; alone, it produces relative time.
* ``start``: ISO 8601 date-time with ``T`` between date and time; with
  ``duration`` or ``end``, it produces absolute time.
* ``end``: ISO 8601 date-time after ``start``.

Exactly one combination is allowed: ``duration`` alone, ``start`` with
``duration``, or ``start`` with ``end``. ``record_step`` and ``duration`` must
be positive ``HH:MM:SS`` strings (for example ``"26:00:00"``); hours may
exceed 23, minutes and seconds must be 0–59. ``end`` must follow ``start``.
For ``start``/``end``, either both timestamps carry timezone offsets or neither does.
Offsets are ignored by GRASS: execution uses the supplied wall-clock values without timezone conversion.
All members share the same time settings.

``input``
---------

These options name a GRASS raster map or space-time raster dataset (STRDS), possibly qualified as ``name@mapset``.
If the name resolves to both a raster and an STRDS, it is ambiguous and rejected.
Maps in another mapset must be on the GRASS search path.
``ground_elevation`` and ``friction`` must contain finite values in the active domain.
Each of the following accepts one string or a non-empty list of map-name strings:

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning (unit)
   * - ``ground_elevation`` (required)
     - Terrain elevation (m).
   * - ``friction`` (required)
     - Manning's roughness ``n`` (s m⁻¹ᐟ³).
   * - ``water_depth``
     - Initial water depth (m).
   * - ``water_surface_elevation``
     - Initial surface elevation, ground elevation plus depth (m).
   * - ``rainfall_rate``
     - Rainfall (mm/h).
   * - ``inflow``
     - Point inflow (m/s).
   * - ``losses``
     - Loss rate (mm/h).
   * - ``boundary_type``
     - Boundary condition type (integer map values).
   * - ``boundary_value``
     - Boundary condition value (m).

``water_depth`` and ``water_surface_elevation`` cannot both be specified.
Boundary types are 0/1 (closed), 2 (open), and 4 (specified water depth);
type 3 is not implemented. Open and closed types only apply at the
computational region border. See :doc:`conf_file` for detailed input and
boundary behavior, including caveats about time-varying input datasets.

``input.infiltration`` is a single mapping (defaults to ``{type: none}``) or
a list of complete mappings, each with one of these shapes:

.. code-block:: yaml

   infiltration:
     - type: "none"
     - type: "constant"
       rate: "infiltration_rate@PERMANENT"
     - type: "green-ampt"
       effective_porosity: "porosity@PERMANENT"
       capillary_pressure: "suction@PERMANENT"
       hydraulic_conductivity: "conductivity@PERMANENT"
       soil_water_content: "initial_moisture@PERMANENT"

``constant.rate`` is an infiltration rate map (mm/h).
The Green-Ampt maps ``effective_porosity`` (fraction),
``capillary_pressure`` (mm),
and ``hydraulic_conductivity`` (mm/h) are required together.
``soil_water_content`` (fraction) is optional.
The model types are mutually exclusive, and fields inside an infiltration alternative cannot be sweeps.

``parameters``
--------------

All options are optional.
Each takes a finite YAML number (integers are accepted) or
a non-empty list of finite numbers.
Quoted numbers are not accepted.
Values must also meet these model limits:

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
     - Default
     - Range
   * - ``hmin``
     - Minimum water depth (m)
     - 0.005
     - ≥ 0
   * - ``cfl``
     - Time-step coefficient
     - 0.7
     - 0.01–1
   * - ``theta``
     - Inertia weighting
     - 0.9
     - 0–1
   * - ``g``
     - Gravity (m/s²)
     - 9.80665
     - ≥ 0
   * - ``dtmax``
     - Maximum surface-flow time-step (s)
     - 5.0
     - > 0
   * - ``dtinf``
     - Infiltration/losses time-step (s)
     - 60.0
     - > 0
   * - ``slope_threshold``
     - GMS slope threshold (m/m)
     - 0.8
     - ≥ 0
   * - ``max_slope``
     - Maximum GMS slope (m/m)
     - 0.8
     - ≥ ``slope_threshold``
   * - ``max_error``
     - Maximum relative volume error
     - 0.05
     - > 0

``drainage``
------------

Optional; enables SWMM coupling.
``swmm_input`` is required when the section is present.
Each supplied option can be a scalar or a non-empty sweep list:

* ``swmm_input``: path to an EPA SWMM ``.inp`` file.
  Relative paths are resolved from the YAML file's directory, not the current working directory.
* ``orifice_coeff``: flow-exchange coefficient, float from 0 to 1 (default 0.167).
* ``free_weir_coeff``: flow-exchange coefficient, float from 0 to 1 (default 0.54).
* ``submerged_weir_coeff``: flow-exchange coefficient, float from 0 to 1 (default 0.056).

Omitted coefficients use the defaults.
SWMM input files must exist at resolution time.
``outputs.drainage`` is optional even when coupling is enabled
Without it, no drainage vector dataset is written.
See :doc:`conf_file` for the contents of drainage vector output.

``outputs``
-----------

This section is required, but any of its four subsections may be omitted:

* ``rasters``: requires ``prefix`` (GRASS name prefix template) and
  ``variables`` (non-empty list of distinct output names). For each variable,
  Itzi writes a STRDS named ``<prefix>_<variable>``. Supported variables are
  ``water_depth``, ``max_water_depth``, ``water_surface_elevation``,
  ``flow_speed``, ``max_flow_speed``, ``flow_velocity_direction``, ``froude``,
  ``flow_rate_x``, ``flow_rate_y``, ``mean_boundary_flow``,
  ``mean_infiltration``, ``mean_rainfall``, ``mean_inflow``, ``mean_losses``,
  ``mean_drainage_flow``, and ``created_volume``. See :doc:`conf_file` for
  meanings, units, and output caveats. Legacy output aliases are not accepted.
* ``statistics``: requires ``file`` (non-empty CSV file path template); the
  file is updated at each ``record_step``. See :doc:`conf_file` for its columns.
* ``drainage``: requires ``vector_dataset`` (GRASS vector dataset name
  template); requires the top-level ``drainage`` section. Drainage vectors
  form a space-time vector dataset (STVDS).
* ``manifest``: requires ``file`` (non-empty YAML path template). By default,
  the manifest goes to ``results/<ensemble-id>.manifest.yaml`` relative to the
  YAML source. See :doc:`manifest` for its contents.

``prefix``, ``statistics.file``, and ``drainage.vector_dataset`` can use
``{ensemble}`` and ``{simulation}`` placeholders. ``manifest.file`` can use
only ``{ensemble}``. Formatting specifications and conversions are unsupported;
use ``{{`` and ``}}`` for literal braces. Filesystem output paths may be
absolute or relative to the YAML source file; ``~`` is expanded. GRASS output
names must be unqualified, valid names in the execution mapset (not ``MASK``).
Derived record names must also be valid. Colliding output names, existing
outputs without ``-o``, and file outputs that would overwrite the YAML source,
SWMM input, or each other are rejected during validation.
