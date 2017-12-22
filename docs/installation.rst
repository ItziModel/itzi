
Installation
============

Availability
------------

The python package for Itzï is on `pypi <https://pypi.python.org/pypi/itzi>`__.
You can browse and download the source code on `bitbucket <https://bitbucket.org/itzi-model/itzi>`__.

Installation on GNU/Linux
-------------------------

Itzï depends on `GRASS GIS 7 <https://grass.osgeo.org/download/>`__ and `NumPy <http://www.numpy.org/>`__.
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

    tar -xvf itzi-17.1.tar.gz
    cd itzi-17.1
    python setup.py install --user

.. note :: For some reason not related to Itzï, pip does not always place the Itzï executable in an accessible place.
    If calling *itzi* returns a *command not found* error, you need to add the installation directory (usually *~/.local/bin*) to your PATH.

Installation for all users
^^^^^^^^^^^^^^^^^^^^^^^^^^

This requires root access.
The steps are the same as above, with the addition of the use of sudo::

    sudo pip install itzi


Installation on Windows
-------------------------

Itzï can be run on Windows 10 using the Windows Subsystem for Linux (WSL).
For that, you'll need at least Windows 10 64bits Anniversary Update.
The Creators Update will save you some steps in the installation.

To install WSL, follow the steps given by `Microsoft <https://msdn.microsoft.com/en-us/commandline/wsl/install_guide>`__.

Once WSL is installed, launch the bash command line and check your ubuntu version::

    lsb_release -a

If you have Ubuntu 14.04, you'll first need to add a repository to have access to GRASS 7.
This step is not necessary if you have Ubuntu 16.04 (installed if you have Windows Creators Update)::

    add-apt-repository ppa:ubuntugis/ppa

You can then install the prerequisites::

    sudo apt-get update
    sudo apt-get install grass-dev grass-core python-pip

Once everything is installed, the installation steps are the same as GNU/Linux.

Verification of the installation
--------------------------------

To check if everything went fine::

    itzi version
    itzi run -h

