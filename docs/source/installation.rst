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

Next we will install the required mathematical libraries:

.. code-block:: bash

    conda install numpy
    conda install scipy

and the pytest library for running the tests:


.. code-block:: bash

    conda install pytest

For now, we use the external library QML to handle the computation of descriptors
for machine learning models. Documentation of QML can be found at https://www.qmlcode.org/.
The simplest way to install QML is with `pip`, 

.. code-block:: bash

    pip install qml --user -U

Note that QML also requires a Fortran compiler which is available from conda.

Running the Tests
-----------------




