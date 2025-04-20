
Programer's manual
==================

Itzï is written principally in Python.
The computationally intensive parts of the code are written in Cython.


Source code management
----------------------

The source code is managed by `git <https://git-scm.com/>`__ and hosted on `GitHub <https://github.com/ItziModel/itzi>`__.
The best way to contribute is to fork the main repository,
make your modifications and then create a pull request on github.
The repository has two branches:

- *master* than contain the current released version.
- *dev* where the main development takes place.

The code should be tested in *dev* before being merged to master for release.
Any larger, possibly breaking changes should be done in a feature branch from *dev*.


Development environment
-----------------------

We use `PDM <https://pdm-project.org>`__ to manage the environment and dependencies.
Once the repository is cloned, you can create a virtual environment and install itzi in editable mode
alongside the dependencies with:

.. code:: sh

    $ pdm install

This will create a virtual environment and install all the dependencies listed in the *pyproject.toml* file.
Now, every change you make to the Python code will be directly reflected when running *itzi* from the command line or the tests.

The default dependency resolver of PDM might fails.
If so, install `uv <https://docs.astral.sh/uv/>`__ and set pdm to use it:

.. code:: sh

    $ pdm config use_uv true


Cython code
-----------

The install script should automatically compile the Cython code.
However, if you want to compile it manually,
you can do so by running the following command in the root directory of the repository:

.. code:: sh

    $ cython -3 src/itzi/flow.pyx
    $ pdm install


Testing
-------

Testing is done with pytest.
Due to global variables in GRASS (`see issue <https://github.com/OSGeo/grass/issues/629>`__),
the tests must be run in separate processes using *pytest-forked*.

.. code:: sh

    $ pdm run pytest --forked -v

To estimate the test coverage:

.. code:: sh

    $ pdm run pytest --cov=itzi --forked -v

Select the python version to test against with *pdm use*.
Test against the 3 `last versions of python <https://devguide.python.org/versions/>`__.
In april 2025, that would be 3.11, 3.12 and 3.13.


Coding style
------------

Code formatting and linting is done with `ruff <https://docs.astral.sh/ruff/>`__.
Formatting is checked automatically before each commit with a pre-commit hook.
pre-commit hooks should be installed after first cloning the repository by following the instructions on the *pre-commit* `official website <https://pre-commit.com/>`__.


Documentation
-------------
The documentation is written in reStructuredText and is built with Sphinx.
It is located in the *docs* directory.
It is automatically built and published on `readthedocs <https://itzi.readthedocs.io>`__.
To build the documentation locally, you first need to install sphinx, along with sphinx-argparse and sphinx_rtd_theme.
Then you can build the documentation locally:

.. code:: sh

    $ cd docs
    $ sphinx-build . _build


Release process
---------------

Once a potential feature branch is merged into *dev*:

- Make sure all the tests pass
- Merge *dev* into *master*
- Bump the version number in the *pyproject.toml* file and the documentation *conf.py*
- Write the release notes
- Update the documentation if necessary
- Run the tests one last time
- Create an annotated tag for version number
- Create the package and push to pypi
- Write a blog post announcing the version
- Post a link to the announcement on the user mailing list
