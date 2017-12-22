
Programer's manual
==================

Itzï is written principaly in Python.
The computationally intensive parts of the code and some C bindings are written in Cython.
Itzï includes the SWMM source code, which is written in C.
Due to incomplete compatibility of GRASS GIS with Python 3, Itzï is meant to be used with Python 2.7 only.
However, the goal is to not use any idiom that is incompatible with Python 3.

We do our best to keep Itzï `PEP8-compliant <https://www.python.org/dev/peps/pep-0008/>`__.
Please use the `pycodestyle <https://pypi.python.org/pypi/pycodestyle/>`__ utility to check your code for compliance.
Sometimes it is difficult to keep the line length under 72 characters.
The line length could be extended to 90 characters in those cases.


Source code management
----------------------

The source code is managed by `git <https://git-scm.com/>`__ and hosted on `Bitbucket <https://bitbucket.org/itzi-model/itzi>`__.
The best way to contribute is to fork the main repository, make your modifications and then create a pull request on Bitbucket.
The repository have two branches:

- *master* than contain the current released verion.
- *dev* where the main development takes place.

The code should be tested in *dev* before being merged to master for release.
No formal test suite exists for now.

Development environment
-----------------------

We recommend to create a virtual environment to work on the source code.
This will prevent to mess with another installed verion of Itzï.

.. code:: sh

    $ virtualenv itzi_dev

Then you can activate the virtualenv and install the dev version of Itzï.

.. code:: sh

    $ source itzi_dev/bin/activate
    $ pip install numpy
    $ cd itzi
    $ pip install -e .

Now, every change you make to the Python code will be directly reflected when running *itzi* from the command line.
To leave the virtualenv:

.. code:: sh

    $ deactivate

Cython code
-----------

After modifying the Cython code, you should first compile it to C, then compile the C code.

.. code:: sh

    $ cython itzi/flow.pyx
    $ rm -rf build/
    $ pip install -e .


Packaging
---------

The process for packaging and sending to pypi is done via a bitbucket pipeline, defined in the bitbucket-pipelines.yml file.
