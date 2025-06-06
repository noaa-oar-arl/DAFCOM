Examples
========

This section provides examples of how to use DAFCOM for various tasks.

Direct XGBoost Example
--------------------

This example demonstrates how to train and use a standalone XGBoost model for direct forecasting:

.. code-block:: bash

   python -m dafcom.forecast.examples.direct_xgboost_example --config path/to/config.yaml

Full example code:

.. literalinclude:: ../src/dafcom/forecast/examples/direct_xgboost_example.py
   :language: python
   :linenos:

Two-Stage Pipeline Example
-----------------------

This example demonstrates how to use the two-stage forecasting pipeline:

.. code-block:: bash

   python -m dafcom.forecast.examples.two_stage_pipeline_example --config path/to/config.yaml

Full example code:

.. literalinclude:: ../src/dafcom/forecast/examples/two_stage_pipeline_example.py
   :language: python
   :linenos:
