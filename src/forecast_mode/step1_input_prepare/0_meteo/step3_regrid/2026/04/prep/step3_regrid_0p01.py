#author:beiming.tang
#date:04/23/2025

import numpy as np
from netCDF4 import Dataset
import xarray as xr
import xesmf as xe

'''
1) read inputs
'''
variable = 'precipitation'
month='04'

dir_data = '/scratch3/NCEPDEV/stmp/Beiming.Tang/Project/Dafcom/Data/0_prepare_Meteo/step1_surfacemap/2026/2026_'+month+'/'
ds = xr.open_dataset(dir_data+'Surface_Map_'+month+'_'+variable+'.nc')

dr = ds[variable][:]

'''
2) define output grid
'''

ds_out = xr.Dataset(
    {
         "lat":(["lat"], np.arange(49,25,-0.01),{"units":"degrees_north"}),
         "lon":(["lon"], np.arange(-125,-65,0.01),{"units":"degrees_east"}),
    }
)

'''
3) regrid
'''
regridder = xe.Regridder(ds,ds_out,'bilinear')
dr_out = regridder(dr, keep_attrs=True)

lat_w = [49 - 0.01* X for X in range(2400)]
lon_w = [-125+0.01* X for X in range(6000)]

'''
4) write into netcdf file
'''

ft = Dataset((variable+'_0p01.nc'),'w',format = 'NETCDF4')
lat = ft.createDimension('latitude',2400)
lon = ft.createDimension('longitude',6000)
time = ft.createDimension('time',len(dr_out))

lat_new = ft.createVariable('lat','f4',('latitude'))
lat_new.units = ''
lat_new.description = ''

lon_new = ft.createVariable('lon','f4',('longitude'))
lon_new.units = ''
lon_new.description = ''

variable_new = ft.createVariable(variable,'f4',('time','latitude','longitude'))
variable_new.units = ''
variable_new.description = ''

lat_new[:] = np.array(lat_w)
lon_new[:] = np.array(lon_w)
variable_new[:,:,:] = np.array(dr_out)

ft.close()
















































