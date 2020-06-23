.. _`sec:running_cliff`:

Running CLIFF
=============

To run CLIFF either with the main script or through the API, be sure to have the main directory added to
your PATH and PYTHONPATH:


.. code-block:: bash

    export PATH=/path/to/code/cliff/cliff:$PATH
    export PYTHONPATH=/path/to/code/cliff:$PYTHONPATH


Quickstart
-----------

The simplest way to run CLIFF is with the command line runscript. CLIFF can also be run using a python API
for more specialized tasks. The runscript requires all monomers to be located in separate .xyz files
in one directory. All other options, including parameters, algorithm choices, etc. are specified in
a config.ini file. The script can be executed by calling:

.. code-block:: bash

    run_cliff.py -i /path/to/config.ini -f /path/to/.xyzs



Setting up config.ini
---------------------


Output options
--------------

