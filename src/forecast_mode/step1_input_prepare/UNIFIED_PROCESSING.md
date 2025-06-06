# Unified Atmospheric Data Processing in DAFCOM

## Overview

The DAFCOM system has been refactored to use a unified approach for processing atmospheric data. This document explains the new approach, its advantages, and how to use it effectively.

## What Has Changed?

Previously, DAFCOM processed different types of atmospheric data (meteorology, chemistry, AOD) separately, requiring different file patterns and processing methods. The new unified approach:

1. **Uses a single file pattern** for all atmospheric data
2. **Groups variables by category** rather than by file type
3. **Processes all variables through a unified interface**
4. **Automatically detects and handles duplicate time dimensions**
5. **Uses pure xarray-based processing** with no mixing of netCDF4 libraries
6. **Tracks variable categories** in metadata for downstream processing
7. **Optimizes memory usage** with improved chunking and parallelization

## Key Benefits

- **Simpler code**: One method to process all atmospheric data instead of multiple type-specific methods
- **Better organization**: Variables are categorized and tagged regardless of their source files
- **Flexible filtering**: Process all data or filter by category or specific variables
- **Backward compatibility**: Legacy methods still work but use the unified approach internally
- **Enhanced metadata**: Each variable carries category and scaling information in its attributes
- **Improved performance**: Unified processing allows better optimization and memory management

## How to Use the Unified Approach

### Basic Usage

```python
# Initialize the loader
from modules.parallel_data_loader import ParallelDataLoader
data_loader = ParallelDataLoader('path/to/config.yaml')

# Process all atmospheric data (all categories)
all_data = data_loader.process_atmospheric_data()

# Process only meteorological variables
meteo_data = data_loader.process_atmospheric_data(category="meteorology")

# Process only chemistry variables
chem_data = data_loader.process_atmospheric_data(category="chemistry")

# Process specific variables across categories
variables = ["t2m", "pm25", "aod_550nm"]  # From different categories
specific_data = data_loader.process_atmospheric_data(variables=variables)
```

### Configuration

The YAML configuration file now uses a single `atmospheric_data` section for all variables, with `category` attributes to distinguish them:

```yaml
variables:
  atmospheric_data:  # Single unified category for all variables
    # Meteorological variable example
    t2m:
      source_name: "tmp2m"
      units: "K"
      description: "2-meter temperature"
      category: "meteorology"  # Category identifier
      scaling:
        method: "standardize"
        params: {}
      transforms: []

    # Chemistry variable example
    pm25:
      source_name: "PM25_TOT"
      units: "ug/m^3"
      description: "PM2.5 total concentration"
      category: "chemistry"  # Different category
      scaling:
        method: "log"
        params:
          offset: 0.1
      transforms: []
```

### File Patterns

The unified approach uses a single file pattern that matches all required data files:

```yaml
input:
  # Unified file pattern for all atmospheric data
  file_pattern: "aqm.t12z.*.f*"  # Matches both physics and chemistry files
```

Alternative patterns are still supported for backward compatibility:

```yaml
input:
  # Alternative patterns for different file types if needed
  file_patterns:
    physics: "aqm.t12z.phy.f*"
    chemistry: "aqm.t12z.chem_sfc.f*"
```

## Advanced Features

### Category-Based Processing

Variables are automatically organized by their category attribute during processing. You can:

1. Process all variables (`process_atmospheric_data()`)
2. Filter by category (`process_atmospheric_data(category="meteorology")`)
3. Select specific variables (`process_atmospheric_data(variables=["t2m", "pm25"])`)

### Enhanced Memory Optimization

The unified processing approach enables better memory optimization:

1. **Improved chunking**: Optimized chunk sizes for different operations
2. **Single file loading**: Reduces duplicate memory usage from multiple file reads
3. **Automatic memory monitoring**: Adjusts processing based on available memory

### Parallel Processing

Parallel processing is enhanced with:

1. **Variable-level parallelism**: Process multiple variables concurrently
2. **File-level parallelism**: Process multiple files simultaneously
3. **Advanced scheduling**: Optimize task scheduling based on dependencies

## Migration Guide

If you're using the old approach, you can migrate to the unified approach with minimal changes:

1. Update your configuration file to use the unified structure
2. Replace calls to type-specific methods with `process_atmospheric_data()`
3. Add category attributes to your variables if not already present

For backward compatibility, the following methods continue to work:

- `process_data_type(data_type)` - Now maps to `process_atmospheric_data(category=data_type)`
- `process_all_data_types()` - Now maps to `process_atmospheric_data()`

## Example

A comprehensive example script is available at:
`/src/forecast_mode/step1_input_prepare/examples/unified_processing_demo.py`

This demonstrates all aspects of the unified processing approach and includes visualization examples.

## Performance Considerations

For best performance with the unified approach:

1. **Optimize chunk sizes** in the configuration based on your data dimensions
2. **Set appropriate memory limits** to prevent out-of-memory errors
3. **Balance parallelism** with available CPU cores and memory
4. **Enable GPU acceleration** for compatible operations if hardware is available

## Troubleshooting

Common issues and solutions:

1. **Memory errors**: Reduce chunk sizes or worker count in the configuration
2. **Missing variables**: Ensure variable names match between config and data files
3. **Slow processing**: Check for appropriate chunking and parallelization settings
4. **Category-related errors**: Ensure all variables have a valid category attribute

## Future Enhancements

Planned improvements to the unified processing system:

1. **Automatic chunk size optimization** based on data characteristics
2. **Enhanced ML integration** with preprocessing pipelines
3. **Dynamic worker allocation** based on task complexity
4. **Real-time monitoring dashboard** for processing progress and performance
