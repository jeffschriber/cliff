.. _`sec:running_cliff`:

Running CLIFF
=============

To run CLIFF either with the main script or through the API, be sure to have the main directory added to
your PATH and PYTHONPATH:


.. code-block:: bash

    export PATH=/path/to/code/cliff/cliff:$PATH
    export PYTHONPATH=/path/to/code/cliff:$PYTHONPATH


Command Line Usage
------------------

The simplest way to run CLIFF is with the command line runscript. CLIFF can also be run using a python API
for more specialized tasks. The runscript requires all monomers to be located in separate .xyz files
in one directory. All other options, including parameters, algorithm choices, etc. are specified in
a config.ini file, though a config.ini file is not needed nor typically recommended. 

The script can be executed by calling `run_cliff.py`:

.. code-block:: bash

    usage: run_cliff.py [-h] [-i INPUT] [-d DIMER] [-a MONA] [-b MONB] [-n NAME]
                        [-p NPROC] [-fr [FRAG]]
    CLIFF: a Component-based Learned Intermolecular Force Field
    
    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Location of input configuration file
      -d DIMER, --dimer DIMER
                            Directory of dimer xyz files, or an individual dimer
                            xyz file
      -a MONA, --monA MONA  Monomer A xyz file
      -b MONB, --monB MONB  Monomer B xyz file
      -n NAME, --name NAME  Output job name
      -p NPROC, --nproc NPROC
                            Number of threads for numpy
      -fr [FRAG], --frag [FRAG]
                            Do fragmentation analysis

The command like script can be called in two contexts: one where the user specifies one or many dimer
.xyz files, and one where the user specifies two monomer xyz files.


Running with dimer `xyz` files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As shown above, a dimer computation can be invoked by passing a path to a folder
containing one or more dimer xyz files using the `-d` flag:

.. code-block:: bash

   run_cliff.py -d path/to/dimer/xyz/files/

A dimer xyz file specifies a single monomer. The first line is the total number of atoms in the dimer,
the second line contains a comma-separated list of the monomer A charge, monomer B charge, and the number of atoms
in monomer A. Crucially, the coordinates for monomer A need to be listed before those of monomer B.
For example, a dimer xyz for two neutral water molecules may look like:

.. code-block:: bash

    6
    0,0,3
    O 0.000000 0.000000  0.000000
    H 0.758602 0.000000  0.504284
    H 0.260455 0.000000 -0.872893
    O 0.000000 2.000000  0.000000
    H 0.758602 2.000000  0.504284
    H 0.260455 2.000000 -0.872893
    

Running with monomer `xyz` files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, CLIFF can be invoked by using monomer xyz files:

.. code-block:: bash

   run_cliff.py -a monomerA.xyz -b monomerB.xyz

A monomer xyz file can specify one or more monomer geometries. In the above invokation,
interaction energies between all monomers in `monomerA.xyz` and all monomers in `monomerB.xyz` are
computed (interaction energies between mononmers defined in the same xyz are not computed).
In specifying a monomer, the total number of atoms is needed, followed by a comma-separated
list where the last field needs to be the monomer's total charge (other info proir to this charge
is allowed, a label or reference energy is often convenient).

To compute interaction energies between one reference water molecule and two different water molecules,
the monomerA.xyz file can be:

.. code-block:: bash

    3
    reference_water,0
    O 0.000000 0.000000  0.000000
    H 0.758602 0.000000  0.504284
    H 0.260455 0.000000 -0.872893

and the monomerB.xyz can be, for example:

.. code-block:: bash

    3
    test_water_1,0
    O 0.000000 2.000000  0.000000
    H 0.758602 2.000000  0.504284
    H 0.260455 2.000000 -0.872893

    3
    test_water_2,0
    O 0.000000 4.000000  0.000000
    H 0.758602 4.000000  0.504284
    H 0.260455 4.000000 -0.872893


Notes on Options
----------------

As mentioned, options including global parameters, model locations, 
can be modified by manually specifying a `config.ini` file.
However, we do not recommend using such a file, as the CLIFF
method is optimized for use with all preset defaults,
and any additional modifications to these defaults
are considered expert options. For interested users, we
point out all available options and defaults are defined
in `cliff/helpers/options.py`.



