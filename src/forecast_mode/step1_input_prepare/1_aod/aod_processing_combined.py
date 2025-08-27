#author: Beiming Tang
#date: Aug 26, 2025

import numpy as np
import os
from netCDF4 import Dataset
import fnmatch
import xarray as xr
import xesmf as xe
import sys

def extract_bev_data(year_list, month_list, day_list, size_lat_ufs, size_lon_ufs):
    """Step 1: Extract BEV data from UFS-AQM output"""
    output_files = []
    
    for year in year_list:
        for month in month_list:
            for day in day_list:
                dir_data = f'/ARLChemWeb/NAQFC/5xpm/{year}/{year}{month}/{year}{month}{day}/'
                pattern = 'aqm.t12z.phy.f*'
                ufs_aqm_list_raw = fnmatch.filter(os.listdir(dir_data), pattern)
                filelist = np.sort(ufs_aqm_list_raw)

                for i in range(24):  # use only the 24 hours forecast
                    filename = dir_data + filelist[i]
                    f = Dataset(filename, 'r')
    
                    lat_w = f.variables['lat'][:]
                    lon_w = f.variables['lon'][:] - 360
                    aod_w = f.variables['aod'][:][0]

                    output_filename = f'BEV_aod_{year}{month}{day}_{filelist[i].replace("aqm.t","").replace(".phy.f0","")}'
                    ft = Dataset(output_filename, 'w', format='NETCDF4')

                    # Create dimensions
                    ft.createDimension('time', None)
                    ft.createDimension('lat', size_lat_ufs)
                    ft.createDimension('lon', size_lon_ufs)

                    # Create variables
                    lat = ft.createVariable('lat', 'f4', ('lat', 'lon'))
                    lat.units = f.variables['lat'].units
                    lat.description = f.variables['lat'].long_name

                    lon = ft.createVariable('lon', 'f4', ('lat', 'lon'))
                    lon.units = f.variables['lon'].units
                    lon.description = f.variables['lon'].long_name

                    aod = ft.createVariable('aod', 'f4', ('time', 'lat', 'lon'))
                    aod.units = ''
                    aod.description = 'total aod @ 550nm'

                    # Assign data
                    lat[:,:] = lat_w
                    lon[:,:] = lon_w
                    aod[0,:,:] = aod_w

                    ft.close()
                    f.close()
                    output_files.append(output_filename)
    
    return output_files

def create_surface_map(input_files, variable_name, size_lat_ufs, size_lon_ufs):
    """Step 2: Create surface map from BEV files"""
    output_filename = f'Surface_Map_{variable_name}.nc'
    
    ft = Dataset(output_filename, 'w', format='NETCDF4')
    time = ft.createDimension('time', len(input_files))
    lat = ft.createDimension('lat', size_lat_ufs)
    lon = ft.createDimension('lon', size_lon_ufs)

    time_new = ft.createVariable('time', 'S1', ('time',))
    time_new.units = ''
    time_new.description = ''

    lat = ft.createVariable('lat', 'f4', ('lat', 'lon'))
    lat.units = ''
    lat.description = ''

    lon = ft.createVariable('lon', 'f4', ('lat', 'lon'))
    lon.units = ''
    lon.description = ''

    variable = ft.createVariable(variable_name, 'f4', ('time', 'lat', 'lon'))
    variable.units = ''
    variable.description = ''

    for i, filename in enumerate(input_files):
        f = Dataset(filename, 'r')
        
        time_w = os.path.basename(filename).replace('BEV_aod_', '')
        lat_w = f.variables['lat'][:]
        lon_w = f.variables['lon'][:]
        variable_w = f.variables[variable_name][0,:]
        
        time_new[i] = time_w
        lat[:,:] = lat_w
        lon[:,:] = lon_w
        variable[i,:,:] = variable_w

        f.close()
    
    ft.close()
    return output_filename

def regrid_to_0p01(input_filename, variable_name, ur_lat,ll_lat, ur_lon, ll_lon):
    """Step 3: Regrid data to 0.01 degree resolution"""
    # Read input data
    ds = xr.open_dataset(input_filename)
    dr = ds[variable_name][:]

    # Define output grid
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(ur_lat, ll_lat, -0.01), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(ll_lon, ur_lon, 0.01), {"units": "degrees_east"}),
        }
    )

    # Perform regridding
    regridder = xe.Regridder(ds, ds_out, 'bilinear')
    dr_out = regridder(dr, keep_attrs=True)
    
    size_lat_regrid = np.abs((ur_lat - ll_lat) / 0.01)
    size_lon_regrid = np.abs((ur_lon - ll_lon) / 0.01)

    lat_w = [ur_lat - 0.01 * x for x in range(size_lat_regrid)]
    lon_w = [ll_lon + 0.01 * x for x in range(size_lon_regrid)]

    # Write regridded data to netCDF file
    output_filename = f'{variable_name}_0p01.nc'
    ft = Dataset(output_filename, 'w', format='NETCDF4')
    
    lat = ft.createDimension('latitude', size_lat_regrid)
    lon = ft.createDimension('longitude', size_lon_regrid)
    time = ft.createDimension('time', len(dr_out))

    lat_new = ft.createVariable('lat', 'f4', ('latitude'))
    lat_new.units = ''
    lat_new.description = ''

    lon_new = ft.createVariable('lon', 'f4', ('longitude'))
    lon_new.units = ''
    lon_new.description = ''

    variable_new = ft.createVariable(variable_name, 'f4', ('time', 'latitude', 'longitude'))
    variable_new.units = ''
    variable_new.description = ''

    lat_new[:] = np.array(lat_w)
    lon_new[:] = np.array(lon_w)
    variable_new[:,:,:] = np.array(dr_out)

    ft.close()
    return output_filename

def main():
    # Configuration
    year_list = ['2025']
    month_list = ['04']
    day_list = ['29', '30']  # Adjust as needed
    variable_name = 'aod'
    size_lat_ufs = 488  # UFS-AQM latitude size
    size_lon_ufs = 775  # UFS-AQM longitude size
    ur_lat = 49.0      # Upper right latitude for regridding
    ll_lat = 25.0      # Lower left latitude for regridding
    ur_lon = -65.0     # Upper right longitude for regridding
    ll_lon = -125.0    # Lower left longitude for regridding
    
    # Step 1: Extract BEV data
    print("Step 1: Extracting BEV data...")
    bev_files = extract_bev_data(year_list, month_list, day_list, size_lat_ufs, size_lon_ufs)
    
    # Step 2: Create surface map
    print("Step 2: Creating surface map...")
    surface_map_file = create_surface_map(bev_files, variable_name, size_lat_ufs, size_lon_ufs)
    
    # Step 3: Regrid to 0.01 degree resolution
    print("Step 3: Regridding to 0.01 degree resolution...")
    final_output = regrid_to_0p01(surface_map_file, variable_name, ur_lat, ll_lat, ur_lon, ll_lon)
    
    print(f"Processing complete. Final output saved to: {final_output}")

if __name__ == "__main__":
    main()
