
Programer's manual
==================

Itzï is written principaly in Python.
The computationally intensive parts of the code and some C bindings are written in Cython.
Itzï includes the SWMM source code, which is written in C.
As of version 20.5, itzi only supports Python 3.

We do our best to keep Itzï `PEP8-compliant <https://www.python.org/dev/peps/pep-0008/>`__.
Please use the `pycodestyle <https://pypi.python.org/pypi/pycodestyle/>`__ utility to check your code for compliance.
Sometimes it is difficult to keep the line length under 72 characters.
The line length could be extended to 90 characters in those cases.


Source code management
----------------------

The source code is managed by `git <https://git-scm.com/>`__ and hosted on `GitHub <https://github.com/ItziModel/itzi>`__.
The best way to contribute is to fork the main repository, make your modifications and then create a pull request on Bitbucket.
The repository have two branches:

- *master* than contain the current released verion.
- *dev* where the main development takes place.

The code should be tested in *dev* before being merged to master for release.
Any larger, possibly breaking changes should be done in a feature branch from *dev*.

Development environment
-----------------------

Create a virtual environment to work on the source code.

.. code:: sh

    $ python3 -m venv itzi_dev

Activate the virtual env and install the dev version of Itzï.

.. code:: sh

    $ source itzi_dev/bin/activate
    $ pip install numpy
    $ cd itzi
    $ pip install -e .

Now, every change you make to the Python code will be directly reflected when running *itzi* from the command line.
To leave the virtual env:

.. code:: sh

    $ deactivate

Cython code
-----------

After modifying the Cython code, you should first compile it to C, then compile the C code.

.. code:: sh

    $ cython -3 itzi/swmm/swmm_c.pyx itzi/flow.pyx
    $ rm -rf build/
    $ pip install -e .


Testing
-------

Testing is done through pytest. Running the tests require the following additional requirements:

- pytest
- pytest-cov
- pytest-xdist
- pandas
- requests

pytest-xdist allows to run each test in a separate process.
To do so, run the following command:

.. code:: sh

    $ pytest --forked -v

To estimate the test coverage:

.. code:: sh

    $ pytest --cov=itzi --forked -v


Release process
---------------

Once a potential feature branch is merged into *dev*:

- Make sure all the tests pass
- Merge *dev* into *master*
- Bump the version number
- Write the release notes
- Update the documentation if necessary
- Run the tests one last time
- Create an annotated tag for version number
- Create the package and push to pypi
- Write a blog post anouncing the version
- Post a link to the anouncement on twitter and the user mailing list
