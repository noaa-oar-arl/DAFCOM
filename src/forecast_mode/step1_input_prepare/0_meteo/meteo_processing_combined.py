# author: Beiming Tang
# date: Aug 27, 2025

import xarray as xr
import numpy as np
import os
import fnmatch
import xesmf as xe
from pathlib import Path

def process_ufs_meteo(year='2025', month='04', day_list=['29', '30'], 
                     variables=['surface_pressure', 't2m', 'specific_humidity_2m', 'evaporation',
                              'u10', 'v10', 'pblh', 'precipitation']):
    """Process UFS meteorological data: extract, combine, and regrid in one step."""
    
    # Step 1: Extract and combine data for all times
    all_data = []
    
    for day in day_list:
        dir_data = f'/ARLChemWeb/NAQFC/5xpm/{year}/{year}{month}/{year}{month}{day}/'
        filelist = np.sort(fnmatch.filter(os.listdir(dir_data), 'aqm.t12z.phy.f*'))
        
        for i in range(24):  # Process 24-hour forecast
            filename = os.path.join(dir_data, filelist[i])
            ds = xr.open_dataset(filename)
            
            # Create time coordinate from filename
            time_str = f"{year}{month}{day}_{filelist[i].replace('aqm.t', '').replace('.phy.f0', '')}"
            
            # Process data for each time step
            data = {
                'lat': ds['lat'],
                'lon': ds['lon'] - 360,  # Convert longitude to -180 to 180
                'time': time_str
            }
            
            # Add variables with their units
            var_mapping = {
                'surface_pressure': ('pressfc', 'Pa'),
                't2m': ('tmp2m', 'K'),
                'specific_humidity_2m': ('spfh2m', 'kg/kg-air'),
                'evaporation': (None, 'W/m^2'),  # Special case - sum of two variables
                'u10': ('ugrd10m', 'm/s'),
                'v10': ('vgrd10m', 'm/s'),
                'pblh': ('hpbl', 'm'),
                'precipitation': ('tprcp', 'kg/m^2')
            }
            
            for var in variables:
                if var == 'evaporation':
                    # Special case for evaporation - sum of two variables
                    data[var] = (ds['evbs_ave'].isel(time=0) + 
                               ds['evcw_ave'].isel(time=0))
                else:
                    src_var = var_mapping[var][0]
                    data[var] = ds[src_var].isel(time=0)
                
            # Create dataset for this time step
            ds_time = xr.Dataset(data)
            all_data.append(ds_time)
            ds.close()
    
    # Combine all times
    combined_ds = xr.concat(all_data, dim='time')
    
    # Add variable attributes
    for var in variables:
        combined_ds[var].attrs['units'] = var_mapping[var][1]
        combined_ds[var].attrs['description'] = var.replace('_', ' ')
    
    # Save combined data
    output_file = f'Surface_Map_combined_{year}{month}.nc'
    combined_ds.to_netcdf(output_file)
    
    # Step 3: Regrid to 0.01 degree resolution (focusing on evaporation as in original)
    if 'evaporation' in variables:
        # Define output grid
        ds_out = xr.Dataset({
            "lat": (["lat"], np.arange(49, 25, -0.01), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(-125, -65, 0.01), {"units": "degrees_east"}),
        })
        
        # Load pre-computed weights if available, otherwise compute new ones
        weight_file = 'bilinear_488x775_2400x6000.nc'
        if os.path.exists(weight_file):
            weight_ds = xr.open_dataset(weight_file)
            regridder = xe.Regridder(combined_ds, ds_out, 'bilinear', weights=weight_ds)
        else:
            regridder = xe.Regridder(combined_ds, ds_out, 'bilinear')
            regridder.to_netcdf(weight_file)
        
        # Regrid evaporation data
        evap_regrid = regridder(combined_ds['evaporation'])
        
        # Create and save regridded dataset
        regridded_ds = xr.Dataset({
            'evaporation': (['time', 'lat', 'lon'], evap_regrid.values),
            'lat': ('lat', ds_out.lat),
            'lon': ('lon', ds_out.lon)
        })
        
        regridded_ds['evaporation'].attrs = combined_ds['evaporation'].attrs
        regridded_ds.to_netcdf(f'evaporation_0p01_{year}{month}.nc')
        
        regridder.clean_weight_file()
    
    return output_file

if __name__ == '__main__':
    # Example usage
    output_file = process_ufs_meteo(
        year='2025',
        month='04',
        day_list=['29', '30']
    )
    print(f"Processing complete. Output saved to: {output_file}")
    print("Regridded evaporation data saved separately if evaporation was included in variables.")
