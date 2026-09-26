Command line usage
==================

Run a simulation
----------------

.. argparse::
   :filename: ../src/itzi/cli_parser.py
   :func: build_parser
   :prog: itzi
   :path: run
    :nodefault:

YAML ensembles
~~~~~~~~~~~~~~

Each YAML document is one ensemble. Use ``--dry`` to perform parsing,
expansion, GRASS input resolution, member-ID calculation, and artifact
validation without writing manifests or model outputs:

.. code-block:: bash

   itzi run --dry studies.yaml

Select resolved members with a qualified ensemble and simulation ID. The
unqualified ID form is accepted only when the batch contains one ensemble.

.. code-block:: bash

   itzi run studies.yaml --member central-city#sim-c1f52e9a

Stage 1 provides basic YAML ensemble execution. YAML hotstart resume mapping
and checkpoint provenance are introduced in Stage 2; the retained INI resume
syntax below continues to apply to legacy INI-only invocations.


Hotstart usage
~~~~~~~~~~~~~~
.. versionadded:: 26.6

Use ``--resume-from`` to resume a run from a hotstart file created by a
previous simulation. The hotstart file does not replace the configuration file:
you still pass the normal Itzi configuration file(s), and Itzi validates the
resumed configuration against the hotstart before starting.

Checkpoint creation and resume-time configuration constraints are documented in
:doc:`conf_file`.
Known restart limitations are summarized in :doc:`faq`.

Single simulation
^^^^^^^^^^^^^^^^^

Resume a single configuration file from one hotstart file:

.. code-block:: bash

   itzi run my_case.ini --resume-from checkpoints/latest_hotstart.zip

Batch mode
^^^^^^^^^^

For batch runs, map each resumed simulation explicitly:

.. code-block:: bash

   itzi run a.ini b.ini \
      --resume-from a.ini=checkpoints/a_hotstart.zip \
      --resume-from b.ini=checkpoints/b_hotstart.zip

Rules
^^^^^

- With a single configuration file, ``--resume-from HOTSTART_PATH`` is valid.
- With multiple configuration files, each ``--resume-from`` value must use the
  ``CONFIG_PATH=HOTSTART_PATH`` form.
- The ``CONFIG_PATH`` part may be either the config path or a basename such as
  ``a.ini``. When basenames are not unique, use the config path.
- At most one hotstart file may be mapped to a given configuration file.
- When multiple ``CONFIG_PATH=HOTSTART_PATH`` mappings are supplied, any batch
  configuration file left unmapped still runs from scratch.


Get the version number
----------------------

.. argparse::
   :filename: ../src/itzi/cli_parser.py
   :func: build_parser
   :prog: itzi
   :path: version
