
Installation
============

Availability
------------

The python package for Itzï is on `pypi <https://pypi.python.org/pypi/itzi>`__.
You can browse and download the source code on `bitbucket <https://bitbucket.org/itzi-model/itzi>`__.

Installation on GNU/Linux
-------------------------

Itzï depends on `GRASS GIS 8.4 or above <https://grass.osgeo.org/download/>`__.
GRASS should therefore be installed in order to use Itzï.
All other dependencies are installed by pip.

To install Itzï, you'll need to have the Python installation software `uv <https://docs.astral.sh/uv>`__ installed.
Install it by following the instructions on the `uv website <https://docs.astral.sh/uv>`__.

Once you have *uv* installed, you can run Itzï using *uvx*.

    uvx itzi

This will automatically install itzi in its own environment and run it.

Installation on Windows
-------------------------

Itzï should be able to run on Windows using the Windows Subsystem for Linux (WSL).
To install WSL, follow the steps given by `Microsoft <https://learn.microsoft.com/en-gb/windows/wsl/install>`__.
Once a GNU/Linux distribution is installed, the installation steps for itzi are the same as above.

Verification of the installation
--------------------------------

To check if everything went fine::

    uvx itzi version
    uvx itzi run -h
