Visualization Guide
=================

DAFCOM provides various visualization capabilities to help you understand your data and evaluate model performance.

Time Series Visualization
----------------------

Visualizing input data, predictions, and actual values:

.. code-block:: python

   import matplotlib.pyplot as plt
   from dafcom.utils.visualization import plot_time_series

   # Plot time series of actual vs predicted values
   plot_time_series(
       dates=test_data.index,
       actual=test_data['target'],
       predicted=predictions,
       title='PM2.5 Forecasting Results',
       ylabel='PM2.5 (μg/m³)'
   )

   # Save the plot
   plt.savefig('forecast_results.png', dpi=300)
   plt.show()

Feature Importance
---------------

Visualizing feature importance from XGBoost models:

.. code-block:: python

   from dafcom.utils.visualization import plot_feature_importance

   # After training a model
   metrics = model.train(train_data)

   # Plot feature importance
   plot_feature_importance(
       feature_names=model.feature_columns,
       importance_values=metrics['feature_importance'],
       title='Feature Importance',
       figsize=(10, 6)
   )
   plt.show()

Error Distribution
---------------

Analyzing error distributions:

.. code-block:: python

   from dafcom.utils.visualization import plot_error_distribution

   # Calculate errors
   errors = test_data[model.target_column].values - predictions

   # Plot error distribution
   plot_error_distribution(
       errors=errors,
       title='Forecast Error Distribution',
       xlabel='Error (μg/m³)'
   )
   plt.show()

Scatter Plots
-----------

Creating scatter plots of predicted vs actual values:

.. code-block:: python

   from dafcom.utils.visualization import plot_scatter

   # Create scatter plot
   plot_scatter(
       actual=test_data[model.target_column].values,
       predicted=predictions,
       title='Predicted vs Actual PM2.5',
       xlabel='Actual PM2.5 (μg/m³)',
       ylabel='Predicted PM2.5 (μg/m³)'
   )

   # Add 1:1 line
   plt.plot([0, 100], [0, 100], 'k--', alpha=0.5)
   plt.show()

Heatmaps
-------

Visualizing correlations between features:

.. code-block:: python

   from dafcom.utils.visualization import plot_correlation_heatmap

   # Plot correlation heatmap
   plot_correlation_heatmap(
       data=train_data[model.feature_columns + [model.target_column]],
       title='Feature Correlation Heatmap'
   )
   plt.show()

Spatial Visualizations
-------------------

If your data includes spatial coordinates, visualize predictions on a map:

.. code-block:: python

   from dafcom.utils.visualization import plot_spatial_forecast

   # Assuming test_data has lat and lon columns
   plot_spatial_forecast(
       latitudes=test_data['lat'],
       longitudes=test_data['lon'],
       values=predictions,
       title='PM2.5 Forecast Map',
       colormap='viridis',
       vmin=0,
       vmax=50
   )
   plt.show()

Customizing Visualizations
-----------------------

All visualization functions accept standard matplotlib parameters:

.. code-block:: python

   # Customize plot appearance
   plot_time_series(
       dates=test_data.index,
       actual=test_data['target'],
       predicted=predictions,
       title='Custom Time Series Plot',
       figsize=(12, 6),
       grid=True,
       linewidth=2,
       markersize=4,
       color_actual='darkblue',
       color_predicted='crimson',
       alpha=0.8
   )

   # Adjust figure layout
   plt.tight_layout()
   plt.show()

Saving Visualizations
------------------

Save plots in various formats:

.. code-block:: python

   # Save as PNG with high resolution
   plt.savefig('output_figure.png', dpi=300, bbox_inches='tight')

   # Save as PDF for publication
   plt.savefig('output_figure.pdf', format='pdf', bbox_inches='tight')

   # Save as SVG for web use
   plt.savefig('output_figure.svg', format='svg', bbox_inches='tight')
