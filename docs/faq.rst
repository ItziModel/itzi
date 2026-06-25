
Frequently Asked Questions
==========================

.. _numerical-instabilities:

Controlling numerical instabilities
-----------------------------------

.. versionadded:: 25.8
    Early detection of numerical instabilities.

In some cases, numerical instabilities could occur, stopping the simulation with a message similar to this one:

.. code:: sh

    WARNING: Error during execution: itzi.itzi_error.MassBalanceError: Mass balance error 0.07 exceeds threshold 0.05

There are two ways to reduce those instabilities.
The first and more effective one is reducing the time-step,
which could be achieved by changing two options:

-  *cfl* that applies to every calculated time-step
-  *dtmax* that defines a maximum value for the time-step

The second one is by reducing the *theta* option.
Please note however that a value below 0.7 could be counter-productive.


Why does a resumed run differ from an uninterrupted run?
--------------------------------------------------------

.. versionadded:: 26.6

Hotstart resume is a continuation mechanism, not a guarantee of bitwise
identity with an uninterrupted run.

For surface-only cases, resumed runs should normally be very close to the
original run. With drainage enabled, the current SWMM hotstart behavior is not
always restart-exact.
A resumed run can therefore show small differences relative to an uninterrupted run
even when the hotstart file and configuration are valid.

See :doc:`conf_file` for the configuration constraints enforced when resuming.


Performances and computer resources usage
-----------------------------------------

*Itzï* is parallelized using OpenMP and makes use of compiler-level vector optimizations.
By default, it will try to use all available hardware threads on the machine.
The number of threads used can be changed by setting the environment variable OMP\_NUM\_THREADS.

Using a computer with more cores and faster RAM will likely decrease the computation time.
Additionally, it is recommended to disable multithreading (SMT) if your CPU support it;
SMT is counterproductive for the type of computing *Itzï* does.

For an example of expected performance with *Itzï* version 26.6, a 24h simulation of urban floods with direct
rainfall on a 5m DEM of 3.5 millions cells takes around 2 hours on a 16 cores AMD Ryzen 5900XT.


How to decrease computation time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The factors that influence the computation time are:

-  The duration of the simulated event.
-  The number of cells in the domain.
-  The number of wet cells in the domain.
   Direct rainfall is more demanding.
-  The cell size. A smaller cell size decreases the time-step duration.
-  The maximum water depth in the domain.
   The higher the water depth, the shorter the time-step.
-  The number and frequency of result maps.
   Disk operations are slow; writing more maps to the disk will slow down the simulation.

As we can see, they are two main categories of factors.
Those that increase the raw computation load (more cells, longer simulations),
and those that lower the simulation time-step.
For the same study area, increasing the cell size is the easiest way to make a simulation run faster,
because it influences both the number of cells and the time-step.


Memory usage
~~~~~~~~~~~~

*Itzï* 26.6 uses around 275 MB of RAM for each million cells in the domain.
