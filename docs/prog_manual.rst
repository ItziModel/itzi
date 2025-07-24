
Programer's manual
==================

Itzï is written principally in Python.
The computationally intensive parts of the code are written in Cython.


Source code management
----------------------

The source code is managed by `git <https://git-scm.com/>`__ and hosted on `GitHub <https://github.com/ItziModel/itzi>`__.
The main repository has only the one branch, the *main* branch.
The best way to contribute is to fork the main repository,
create a specific branch for your changes, then create a pull request on github.

Development environment
-----------------------

We use `uv <https://docs.astral.sh/uv/getting-started/installation/>`__ to manage the environment and dependencies.
Once the itzi repository is cloned and uv installed, you can run itzi with:

.. code:: sh

    uv run itzi

This will create a virtual environment, install all the dependencies listed in the *pyproject.toml* file, and build the Cython extensions.
Now, every change you make to the Python code will be directly reflected when running the tests or *uv run itzi* .


Cython code
-----------

After editing the Cython code, you need to compile it again.
You can do so by running the following command in the root directory of the repository:

.. code:: sh

    uv pip install -e .


Testing
-------

Testing is done with pytest.
Due to global variables in GRASS (`see issue <https://github.com/OSGeo/grass/issues/629>`__),
the tests must be run in separate processes using *pytest-forked*.

.. code:: sh

    uv run pytest --forked -v tests/

To estimate the test coverage:

.. code:: sh

    uv run pytest --cov=itzi --forked -v tests/

Select the python version to test against with the *--python* option.
For example *uv run --python 3.12 pytest tests/* for python 3.12.
This will automatically install the correct python version.
Test against the 3 `last versions of python <https://devguide.python.org/versions/>`__.


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
To build the documentation locally, you first need to install *sphinx*, along with *sphinx-argparse* and *sphinx_rtd_theme*.
You can then build the documentation locally:

.. code:: sh

    cd docs
    sphinx-build . _build


Continuous integration
----------------------

Tests are automatically run with GitHub Actions.
Before committing changes to the workflows, test them locally using `act <https://nektosact.com/>`__.


Release process
---------------

- Make sure all the tests pass
- Bump the version number in the *pyproject.toml* file and the documentation *conf.py*
- Write the release notes
- Update the documentation if necessary
- Run the tests one last time
- Create an annotated tag for version number
- Create the package using the Build CI workflow
- Test the package locally
- Push to pypi
- Write a blog post announcing the version
- Post a link to the announcement on the user mailing list and social networks
