CLIFF
==============================
[//]: # (Badges)
[![codecov](https://codecov.io/gh/jeffschriber/CLIFF/branch/master/graph/badge.svg?token=vYYLLXHWhK)](https://codecov.io/gh/jeffschriber/CLIFF)

Component-based Learned Intermolecular Force Field

# Quick-start guide
## Installation and Dependencies
To handle the dependencies in CLIFF, it is recommended to make a conda environment,

    conda create --name cliff python=3.6

Next we will install the required libraries:


    conda install numpy
    conda install scipy
    conda install qcelemental -c conda-forge

and the pytest library for running the tests:
    
    conda install pytest
    
For now, we use the external library QML to handle the computation of descriptors for machine learning models. Documentation of QML can be found at https://www.qmlcode.org/. The simplest way to install QML is with pip,

    pip install qml --user -U
Note that QML also requires a Fortran compiler which is available from conda.
As a last piece of setup, all KRR models in the three subdirectories in cliff/models/large need to be un-tarred.


# Running the Code
CLIFF can be run using either a provided python script for command-line use, or by using import cliff in user-written python scripts.

## Command-line usage
The simplest way to run CLIFF is with the provided run_cliff.py script, which includes a number of options:

```
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
```
The `-i` flag allows the user to use their own config.ini file to specify any non-default parameters, and is considered more of an expert option.

This script can be called in two contexts, one where the user specifies one or many dimer .xyz files, where the last field of the comma-separated comment line in the xyz specifies the number of atoms in the first monomer. A single dimer .xyz file or an entire directory of such files can be specified with the -d flag,

```
run_cliff.py -d path/to/dimer/xyzs/
```

See `/tests/test_cliff.py` for an example, and see `/tests/dimer_data` for example dimer xys files.

Alternatively, `run_cliff.py` can be called by specifying monomer xyz files. In this approach, two files need to be specified, one for each monomer, and these files contain xyz coordinates for one or more monomers. See /tests/monomer_data for examples. The script can then be run as
```
run_cliff.py -a monomerA.xyz -b monomerB.xyz
```
where all combinations of monomers in monomerA.xyz and monomers in monomerB.xyz are used to compute interaction energies.

When using the runscript, all data is output to a .csv file and to a .log file, the name of which can be specified with `-n`.

## Python usage
CLIFF can also be run in a python script, also using either dimer xyz files or two monomer xyz files. Here, we provide an example script to run CLIFF calculations on the dimer xyz files provided in the test directory:

```
import cliff
import glob

# put your own path before /tests
dimer_xyz = glob.glob("/tests/monomer_data/*.xyz")

# Call a function to load the dimers
# These dimers are QCelemental Molecule objects, with fragments specified
# A list of these objects can also be made directly with QCelementa and passed to CLIFF
dimers = [cliff.load_dimer_xyz(f) for f in dimer_xyz]

# get the energies
energies = cliff.predict_from_dimers(dimers)
```

In the above code, the `energies` object is a list of numpy arrays, with the entire data structure having dimension `# dimers * 4`. The output energies are in kcal/mol and ordered as electrostatics, exchange, induction, and dispersion.

Similarly, we can compute energies using monomer xyzs:

```
import cliff
import glob

# put your own path before /tests
monomerA = "/tests/monomer_data/monomerA.xyz"
monomerB = "/tests/monomer_data/monomerB.xyz"

# Create the qcelemental Molecule objects
monA = cliff.load_monomer_xyz(monomerA)
monB = cliff.load_monomer_xyz(monomerB)

# get the energies
energies = cliff.predict_from_monomer_list(monA,monB)
```

Here, the energies will be an array of lenght of `# monomer A x # monomer B`. For a given indices monA and monB (i.e., their place in the corresponding xyz file), the index of its energy components in energies would be `monA*# monomerA + monB`.




### Copyright

Copyright (c) 2019, Jeff Schriber


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
