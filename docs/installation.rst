
Installation
============

Availability
------------

The python package for Itzï is on `pypi <https://pypi.python.org/pypi/itzi>`__.
You can browse and download the source code on `bitbucket <https://bitbucket.org/itzi-model/itzi>`__.

Installation on GNU/Linux
-------------------------

Itzï depends on `GRASS GIS 7.8 or above <https://grass.osgeo.org/download/>`__ and `NumPy <http://www.numpy.org/>`__.
GRASS should therefore be installed in order to use Itzï.
NumPy is normally installed along GRASS.
All other dependencies are installed by pip.

To install Itzï, you'll need to have the Python installation software *pip* installed.
On Ubuntu, the package is called *python-pip* and is installed as follow::

    sudo apt-get install python-pip

Installation for a single user
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is useful when you do not have root access on the computer.

To download and install the last version of Itzï using pip::

    pip install itzi --user


If Itzï is already installed and you want to update it to the last version::

    pip install itzi --user --upgrade


If you prefer to download and install Itzï manually, you can do it that way::

    tar -xvf itzi-18.2.tar.gz
    cd itzi-18.2
    python setup.py install --user

.. note :: For a reason not related to Itzï, pip does not always place the Itzï executable in an accessible place.
    If calling *itzi* returns a *command not found* error, you need to add the installation directory (usually *~/.local/bin*) to your PATH.

Installation for all users
^^^^^^^^^^^^^^^^^^^^^^^^^^

This requires root access.
The steps are the same as above, with the addition of the use of sudo::

    sudo pip install itzi


Installation on Windows
-------------------------

Itzï can be run on Windows 10 using the Windows Subsystem for Linux (WSL).
For that, you'll need at least Windows 10 64bits Creators Update.

To install WSL, follow the steps given by `Microsoft <https://docs.microsoft.com/en-gb/windows/wsl/install-win10>`__.

You can then install the prerequisites::

    sudo apt-get update
    sudo apt-get install grass-dev grass-core python-pip

Once everything is installed, the installation steps are the same as GNU/Linux.

Verification of the installation
--------------------------------

To check if everything went fine::

    itzi version
    itzi run -h

