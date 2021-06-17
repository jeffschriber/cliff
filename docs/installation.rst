.. _`sec:installation`:

Installation
============

Source Code
-----------

To obtain the CLIFF source code, clone from the main repository with

.. code-block:: bash

    git clone https://github.com/jeffschriber/cliff.git



Dependencies
------------
To handle the dependencies in CLIFF, it is recommended to make a conda environment, 

.. code-block:: bash

    conda create --name cliff python=3.6

Next we will install the required libraries:

.. code-block:: bash

    conda install numpy
    conda install scipy
    conda install qcelemental

and the pytest library for running the tests:


.. code-block:: bash

    conda install pytest

For now, we use the external library QML to handle the computation of descriptors
for machine learning models. Documentation of QML can be found at https://www.qmlcode.org/.
The simplest way to install QML is with `pip`, 

.. code-block:: bash

    pip install qml --user -U

Note that QML also requires a Fortran compiler which is available from conda.
As a final piece of setup, all Kernel-Ridge Regression model files located in
`cliff/models/large` need to be untarred.

Running the Tests
-----------------

Once CLIFF and all dependencies are obtained, it is recommended to run the test suite.
This can be done by going into the tests directory and calling pytest:

.. code-block:: bash

    cd cliff/tests
    pytest





