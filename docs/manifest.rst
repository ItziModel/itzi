Ensemble manifest
=================

Each YAML ensemble run writes a YAML manifest describing its resolved members,
their planned outputs, and their run status. By default, it is saved
as ``results/<ensemble-id>.manifest.yaml`` beside the source YAML file. Set
``outputs.manifest.file`` in the ensemble configuration to choose another file:

.. code-block:: yaml

   outputs:
     manifest:
       file: "results/{ensemble}.yaml"

The manifest path may be absolute or relative to the source YAML file; only
``{ensemble}`` is allowed as a template placeholder (not ``{simulation}``).
An existing manifest requires ``itzi run -o`` to be replaced; if the manifest
destination fails validation, no manifest is written. A dry run validates the
destination but does not write a manifest. An ensemble with no selected resolved
members does not write one, unless it contains unresolved candidates in a run
without ``--member``.

Format
------

The manifest is a YAML mapping with these top-level fields:

* ``manifest_version``: manifest format version, currently ``1``. This is
  distinct from the configuration's ``schema_version``.
* ``last_updated_at``: last write time as an ISO 8601 string with a local
  timezone offset. It changes when the manifest is updated.
* ``ensemble``: ``id`` and ``description`` from the configuration;
  ``description`` is ``null`` when omitted.
* ``source``: absolute ``path`` to the source YAML file and zero-based
  ``document_index`` within that YAML stream.
* ``members``: resolved members in sweep expansion order, followed by
  unresolved candidates in sweep expansion order.

For example, a completed member might appear as follows (names and timings
are illustrative):

.. code-block:: yaml

   manifest_version: 1
   last_updated_at: '2026-09-30T10:20:30+02:00'
   ensemble:
     id: central-city
     description: Central city rainfall study
   source:
     path: /srv/studies/city.yaml
     document_index: 0
   members:
   - simulation_id: sim-1234abcd
     selected: true
     status: completed
     elapsed_seconds: 12.5
     coordinates:
       parameters.cfl: 0.5
     artifacts:
       rasters:
         water_depth: central_city_sim-1234abcd_water_depth@itzi
       drainage: null
       statistics: /srv/studies/results/sim-1234abcd.csv

Member fields
-------------

``simulation_id`` is the resolved member ID (``sim-...``), or ``null`` if
input resolution failed before an ID could be assigned. ``selected`` indicates
whether this member was selected to run. ``coordinates`` maps swept configuration
paths (for example, ``parameters.cfl`` or ``input.rainfall_rate``) to their
chosen values; it is ``{}`` without sweeps. An ``input.infiltration`` sweep
value is a mapping containing its ``type`` and applicable fields.

Resolved members also have ``artifacts``: ``rasters`` maps requested output
variables to GRASS space-time raster dataset IDs (``name@mapset``),
``drainage`` is a GRASS space-time vector dataset ID or ``null``, and
``statistics`` is an absolute CSV file path or ``null``. ``rasters`` is ``{}``
when no raster outputs are requested. These are intended destinations; a
failed or unfinished member may not have produced them. Unresolved candidates
have no ``artifacts`` field.

``elapsed_seconds`` is ``null`` until execution finishes, then records the
elapsed wall-clock time in seconds, including when execution fails. The
``status`` field can be:

* ``not_selected``: excluded by ``--member``.
* ``planned``: selected, but not yet started.
* ``running``: execution started; the manifest is updated before and after
  each member runs.
* ``completed``: execution finished successfully.
* ``validation_failed``: the member could not be run.
* ``execution_failed``: execution raised an error.

Failures also have a ``failure`` mapping with ``phase`` and ``detail``.
Possible phases are ``input_resolution`` (no resolved ID or artifacts),
``artifact_validation`` (conflicting or invalid outputs),
``run_validation`` (pre-run checks), and ``execution``. Unresolved candidates
are marked ``validation_failed`` when running without ``--member``; with
selectors they appear as ``not_selected`` but still carry their failure
detail. A failed YAML document (parse, schema, or expansion error) has no
manifest entry because no ensemble was expanded from it.
