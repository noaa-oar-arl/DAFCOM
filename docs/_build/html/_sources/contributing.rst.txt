Contributing
===========

We welcome contributions to DAFCOM! This document outlines the process for contributing to the project.

Setup Development Environment
---------------------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/your-username/DAFCOM.git
      cd DAFCOM

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Create a branch for your feature:

   .. code-block:: bash

      git checkout -b feature-name

Code Style
---------

We follow PEP 8 for Python code style. Please ensure your code conforms to this standard.

You can check your code style with:

.. code-block:: bash

   flake8 src/dafcom tests

Documentation
------------

All public functions, classes, and methods should have docstrings. We follow the Google style for docstrings:

.. code-block:: python

   def function(arg1, arg2):
       """Summary line.

       Extended description of function.

       Args:
           arg1: Description of arg1
           arg2: Description of arg2

       Returns:
           Description of return value

       Raises:
           ValueError: If arg1 is invalid
       """

Testing
------

All new features should include tests. We use pytest for testing.

Run the tests with:

.. code-block:: bash

   python run_tests.py --pytest

Pull Request Process
------------------

1. Update the README.md and documentation with details of changes
2. Update the version number in pyproject.toml
3. Ensure all tests pass
4. Submit a pull request

Code of Conduct
-------------

Please note that the DAFCOM project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.
