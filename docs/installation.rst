Installation
============

Requirements
-----------

DAFCOM requires Python 3.8 or later and the following packages:

* numpy
* pandas
* matplotlib
* scikit-learn
* xgboost
* tensorflow (for LSTM models)
* pyyaml (for configuration files)

From Source
----------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/your-username/DAFCOM.git
   cd DAFCOM
   pip install -e .

Using pip
--------

DAFCOM can be installed using pip:

.. code-block:: bash

   pip install dafcom

Development Installation
-----------------------

For development purposes, install with all development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Verifying Installation
---------------------

To verify the installation, run:

.. code-block:: bash

   python -c "import dafcom; print(dafcom.__version__)"
