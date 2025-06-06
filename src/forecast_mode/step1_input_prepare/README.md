# DAFCOM Parallel DataLoader

A highly configurable, parallelizable data processing system for atmospheric model data. This system replaces the original 3-step process with a unified, efficient pipeline designed for seamless integration with TensorFlow/Keras ML workflows.

## Features

### 🚀 **High Performance**
- **Parallel Processing**: Uses Dask for distributed computing across multiple workers
- **Memory Efficient**: Chunked processing with configurable memory limits
- **Lazy Loading**: Only loads data when needed, reducing memory footprint
- **Caching**: Intelligent caching system for frequently accessed data

### 🔧 **Highly Configurable**
- **YAML Configuration**: All parameters controlled through a single configuration file
- **Variable Selection**: Choose exactly which variables to process
- **Flexible Time Ranges**: Process any time period with configurable limits
- **Custom Transformations**: Built-in scaling, normalization, and transformation options

### 🧠 **ML-Ready**
- **TensorFlow Integration**: Direct integration with TensorFlow/Keras pipelines
- **Data Generators**: Memory-efficient data generators for training
- **Automatic Scaling**: Built-in scaling and normalization for ML workflows
- **Feature Engineering**: Automatic creation of temporal and spatial features

### 🌍 **Atmospheric Data Optimized**
- **Multi-Modal Support**: Handles meteorology, AOD, and chemistry data
- **Coordinate Handling**: Automatic longitude adjustments and grid management
- **Quality Control**: Built-in data validation and quality checks
- **Interpolation**: Automatic regridding to target resolutions

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For TensorFlow integration (optional)
pip install tensorflow>=2.8.0
```

### Basic Usage

```python
from modules.parallel_data_loader import load_and_process_data

# Simple one-line data loading
datasets = load_and_process_data('config/data_loader_config.yaml')

# Process specific data types
datasets = load_and_process_data(
    'config/data_loader_config.yaml',
    data_types=['meteo', 'chemistry']
)
```

### Advanced Usage

```python
from modules.parallel_data_loader import ParallelDataLoader

# Full control over the loading process
with ParallelDataLoader('config/data_loader_config.yaml') as loader:
    # Process specific variables
    meteo_data = loader.process_data_type(
        'meteo',
        variables=['t2m', 'surface_pressure', 'wind_speed']
    )

    # Save processed data
    loader.save_datasets({'meteo': meteo_data})

    # Create ML-ready datasets
    features, targets = loader.create_ml_dataset({'meteo': meteo_data})
```

### TensorFlow/Keras Integration

```python
from modules.tf_integration import create_atmospheric_dataset, AtmosphericModelBuilder

# Create data generators for training
train_gen, val_gen = create_atmospheric_dataset(
    config_path='config/data_loader_config.yaml',
    data_types=['meteo', 'aod', 'chemistry'],
    target_variables=['pm25', 'o3'],
    batch_size=32,
    sequence_length=24
)

# Build a model
model_builder = AtmosphericModelBuilder({})
model = model_builder.build_lstm_model(
    input_shapes={k: v.shape[1:] for k, v in train_gen[0][0].items()},
    num_outputs=train_gen[0][1].shape[1]
)

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit(train_gen, validation_data=val_gen, epochs=50)
```

## Configuration

The system is controlled through a YAML configuration file. Here's a minimal example:

```yaml
# Global settings
global:
  n_workers: 8
  use_distributed: true
  chunk_size:
    time: 10
    lat: 244
    lon: 388

# Input data
input:
  data_dir_pattern: "/path/to/data/{year}/{year}{month:02d}/{year}{month:02d}{day:02d}/"
  time_range:
    years: [2025]
    months: [4, 5]
    days: [29, 30]
    hours_limit: 24

  file_patterns:
    meteo: "aqm.t12z.phy.f*"
    chemistry: "aqm.t12z.chem_sfc.nc"

# Variables to process
variables:
  meteo:
    t2m:
      source_name: "tmp2m"
      units: "K"
      scaling:
        method: "standardize"

    wind_speed:
      source_name: ["ugrd10m", "vgrd10m"]
      operation: "wind_speed"
      scaling:
        method: "standardize"

# Output grid
output_grid:
  lat_range: [25, 49]
  lon_range: [-125, -65]
  resolution: 0.01
  interpolation:
    method: "linear"
```

## Architecture

### Key Components

1. **ParallelDataLoader**: Main class that orchestrates the entire pipeline
2. **ScalingManager**: Handles all data transformations and scaling for ML
3. **DataTransforms**: Collection of atmospheric data transformations
4. **AtmosphericDataGenerator**: TensorFlow-compatible data generator
5. **AtmosphericModelBuilder**: Pre-configured model architectures

### Data Flow

```
Raw NetCDF Files → Parallel Loading → Variable Extraction →
Quality Control → Grid Interpolation → Scaling/Transforms →
Output (NetCDF/ML-ready arrays)
```

### Parallelization Strategy

- **File-level parallelism**: Different files processed simultaneously
- **Variable-level parallelism**: Multiple variables extracted in parallel
- **Chunk-based processing**: Large datasets split into manageable chunks
- **Distributed computing**: Optional cluster computing with Dask

## Comparison with Original System

| Feature | Original System | New System |
|---------|----------------|------------|
| **Steps** | 3 separate scripts | Single unified pipeline |
| **Libraries** | Mixed netCDF4/xarray | Pure xarray |
| **Configuration** | Hardcoded parameters | YAML configuration |
| **Parallelization** | Limited | Full parallel processing |
| **Memory Usage** | High (loads all data) | Efficient (chunked loading) |
| **ML Integration** | Manual preprocessing | Direct TF/Keras integration |
| **Extensibility** | Difficult | Highly modular |
| **Performance** | Sequential processing | Parallel + distributed |

## Examples

See the `examples/` directory for complete usage examples:

- `run_examples.py`: Comprehensive examples of all features
- Basic usage, ML integration, custom configurations

Run examples:
```bash
python examples/run_examples.py --example basic
python examples/run_examples.py --example ml
python examples/run_examples.py --example all
```

## Performance Tuning

### Memory Optimization

```yaml
global:
  chunk_size:
    time: 10      # Smaller chunks for limited memory
    lat: 100
    lon: 100
  memory_limit_gb: 2  # Limit per worker
```

### Parallel Processing

```yaml
global:
  n_workers: 16         # More workers for more cores
  use_distributed: true # Use cluster if available
  scheduler_address: "tcp://scheduler:8786"  # Cluster scheduler
```

### I/O Optimization

```yaml
output:
  compression:
    compression_level: 6  # Higher compression
    shuffle: true         # Better compression ratios
  output_chunks:
    time: 24             # Optimize for time-series access
    lat: 300
    lon: 750
```

## ML-Specific Features

### Automatic Feature Engineering

```yaml
ml_settings:
  feature_engineering:
    create_temporal_features: true  # Hour, day_of_year, etc.
    create_spatial_derivatives: true  # Gradients, Laplacians
    rolling_statistics:
      windows: [3, 6, 12]  # Rolling means, stds
      stats: ["mean", "std", "min", "max"]
```

### Data Scaling

```yaml
variables:
  meteo:
    t2m:
      scaling:
        method: "standardize"  # or "minmax", "robust", "log"

    precipitation:
      scaling:
        method: "log"
        params:
          offset: 0.01  # Handle zeros
```

### Quality Control

```yaml
quality_control:
  valid_ranges:
    t2m: [200, 330]  # Kelvin
    surface_pressure: [80000, 110000]  # Pa
  out_of_range_action: "mask"  # or "clip", "drop"
```

## Extending the System

### Adding New Variables

```yaml
variables:
  meteo:
    your_variable:
      source_name: "source_var_name"
      units: "units"
      description: "Description"
      scaling:
        method: "standardize"
      transforms: []  # Custom transform functions
```

### Custom Transformations

```python
# In DataTransforms class
@staticmethod
def your_transform(data: xr.DataArray) -> xr.DataArray:
    # Your transformation logic
    return transformed_data
```

### Custom Operations

```yaml
variables:
  meteo:
    combined_var:
      source_name: ["var1", "var2"]
      operation: "your_operation"  # Implement in DataTransforms
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `chunk_size` and `n_workers`
2. **File not found**: Check `data_dir_pattern` and file paths
3. **Variable not found**: Verify `source_name` matches NetCDF variable names
4. **Slow performance**: Increase `n_workers` or enable `use_distributed`

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Check configuration
with ParallelDataLoader('config.yaml') as loader:
    print(loader.config)  # Print full configuration
```

### Performance Monitoring

```yaml
logging:
  level: "DEBUG"
  log_performance: true
  log_memory_usage: true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the DAFCOM system. See the main repository for license information.

## Support

For questions and support:
- Check the examples in `examples/`
- Review the configuration options in `config/data_loader_config.yaml`
- Enable debug logging for detailed information
- Check the troubleshooting section above
