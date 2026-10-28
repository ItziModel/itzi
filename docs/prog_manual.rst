
Programer's manual
==================

Itzï is written in Python.


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

This will create a virtual environment, install all the dependencies listed in the *pyproject.toml* file, including dev dependencies.
Now, every change you make to the Python code will be directly reflected when running the tests or *uv run itzi* .

Testing
-------

Testing is done with pytest.
GRASS 8.5 or above is required. The full suite can run in one pytest process;
GRASS fixtures reuse a session and restore pristine mapset data between tests.
BMI fixtures provide a fresh model for each test.

.. code:: sh

    uv run pytest -v tests/

To estimate the test coverage:

.. code:: sh

    uv run pytest --cov=itzi -v tests/

The tests not relying on GRASS can be run directly:

.. code:: sh

    uv run pytest tests/cli


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
The packages *sphinx*, *sphinx-argparse*, and *sphinx_rtd_theme* are needed to build the docs locally.
They are normally installed automatically with the rest of the dev dependencies.
You can then build the documentation:

.. code:: sh

    cd docs
    uv run sphinx-build . _build


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
- Update and upload the docker image
- Write a blog post announcing the version
- Post a link to the announcement on the user mailing list and social networks
