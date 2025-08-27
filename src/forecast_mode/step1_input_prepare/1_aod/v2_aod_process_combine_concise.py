import numpy as np
import os
import xarray as xr
import xesmf as xe
import fnmatch
from netCDF4 import Dataset

def extract_bev_data(year_list, month_list, day_list, size_lat_ufs, size_lon_ufs):
    """Step 1: Extract BEV data and return as xarray Dataset"""
    all_data = []
    times = []
    
    for year in year_list:
        for month in month_list:
            for day in day_list:
                dir_data = f'/ARLChemWeb/NAQFC/5xpm/{year}/{year}{month}/{year}{month}{day}/'
                pattern = 'aqm.t12z.phy.f*'
                filelist = np.sort(fnmatch.filter(os.listdir(dir_data), pattern))

                for i in range(24):
                    with Dataset(dir_data + filelist[i], 'r') as f:
                        lat = f.variables['lat'][:]
                        lon = f.variables['lon'][:] - 360
                        aod = f.variables['aod'][:][0]
                        
                        time_str = f"{year}{month}{day}_{filelist[i].replace('aqm.t','').replace('.phy.f0','')}"
                        times.append(time_str)
                        all_data.append(aod)

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'aod': (['time', 'lat', 'lon'], np.stack(all_data)),
            'lat': (['lat', 'lon'], lat),
            'lon': (['lat', 'lon'], lon)
        },
        coords={
            'time': times
        }
    )
    return ds

def regrid_to_0p01(ds, variable_name, ur_lat, ll_lat, ur_lon, ll_lon):
    """Regrid data to 0.01 degree resolution"""
    # Define output grid
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(ur_lat, ll_lat, -0.01)),
            "lon": (["lon"], np.arange(ll_lon, ur_lon, 0.01))
        }
    )

    # Perform regridding
    regridder = xe.Regridder(ds, ds_out, 'bilinear')
    dr_out = regridder(ds[variable_name])

    # Create final dataset with regridded data
    final_ds = xr.Dataset(
        {
            variable_name: (['time', 'latitude', 'longitude'], dr_out.values),
            'lat': ('latitude', np.arange(ur_lat, ll_lat, -0.01)),
            'lon': ('longitude', np.arange(ll_lon, ur_lon, 0.01))
        },
        coords={
            'time': ds.time
        }
    )
    
    return final_ds

def main():
    # Configuration
    year_list = ['2025']
    month_list = ['04']
    day_list = ['29', '30']
    variable_name = 'aod'
    size_lat_ufs = 488
    size_lon_ufs = 775
    ur_lat = 49.0
    ll_lat = 25.0
    ur_lon = -65.0
    ll_lon = -125.0
    
    # Extract and process data
    print("Extracting and processing data...")
    ds = extract_bev_data(year_list, month_list, day_list, size_lat_ufs, size_lon_ufs)
    
    # Regrid and save final output
    print("Regridding to 0.01 degree resolution...")
    final_ds = regrid_to_0p01(ds, variable_name, ur_lat, ll_lat, ur_lon, ll_lon)
    
    # Save final output
    output_file = f'{variable_name}_0p01.nc'
    print(f"Saving final output to {output_file}...")
    final_ds.to_netcdf(output_file)
    print("Processing complete.")

if __name__ == "__main__":
    main()
