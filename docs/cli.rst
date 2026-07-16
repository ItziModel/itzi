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
