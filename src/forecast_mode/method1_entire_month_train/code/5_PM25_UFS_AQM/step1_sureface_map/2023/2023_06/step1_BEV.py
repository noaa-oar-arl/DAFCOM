import numpy as np
import os
#import sys
from netCDF4 import Dataset
import fnmatch

'''
1)read data from WRF-Chen output
'''
dir_data ='/groups/ESS3/btang6/DATA/Model_data/UFS_AQM_data/Cmaq54_OriRave1_AfterEmisBugFix/2023_06/'
pattern ='dyn.f*'

wrf_list_raw = fnmatch.filter(os.listdir(dir_data),pattern)
filelist = np.sort(wrf_list_raw)

for i in range(len(filelist)):
    filename = dir_data+filelist[i]
    f = Dataset(filename,'r')

    lat_w = f.variables['lat'][:]
    lon_w = f.variables['lon'][:]-360
    # pressure_w = f.variables['pressfc'][:][0]    #unit = Pa
    # temp_w = f.variables['tmp'][:][0][-1]    #unit = K
    # u_w = f.variables['ugrd'][:][0][-1]
    # v_w = f.variables['vgrd'][:][0][-1]
    
    vmro3_w = f.variables['o3_ave'][:][0][-1]
    #vmrco_w = 10**(3)*f.variables['co'][:][0][-1]
    #vmrno_w = f.variables['no_ave'][:][0][-1]
    vmrno2_w= f.variables['no2_ave'][:][0][-1]
    pm25_w = f.variables['pm25_ave'][:][0][-1]
    
    # air_density = 28.97* pressure_w/(8.314*temp_w)   #unit = g/m^3    
    # mmrbc_w = 10**(-3)*(f.variables['bc_a01'][:][0][0]+f.variables['bc_a02'][:][0][0]+f.variables['bc_a03'][:][0][0])*air_density
    # mmrnit_w = 10**(-3)*(f.variables['no3_a01'][:][0][0]+f.variables['no3_a02'][:][0][0]+f.variables['no3_a03'][:][0][0])*air_density # ug/kg dry air ~ ug/m^3
    # mmrso4_w = 10**(-3)*(f.variables['so4_a01'][:][0][0]+f.variables['so4_a02'][:][0][0]+f.variables['so4_a03'][:][0][0])*air_density
    # mmrnh4_w = 10**(-3)*(f.variables['nh4_a01'][:][0][0]+f.variables['nh4_a02'][:][0][0]+f.variables['nh4_a03'][:][0][0])*air_density
    
    # mmrbc_w =  f.variables['aeci'][:][0][-1]+f.variables['aecj'][:][0][-1] #unit = ug/kg-air
    # mmrnit_w = f.variables['ano3i'][:][0][-1]+f.variables['ano3j'][:][0][-1] #unit = ug/kg-air
    # mmrso4_w = f.variables['aso4i'][:][0][-1]+f.variables['aso4j'][:][0][-1] #unit = ug/kg-air
    # mmrnh4_w = f.variables['anh4i'][:][0][-1]+f.variables['anh4j'][:][0][-1] #unit = ug/kg-air
    # mmrsoil_w = f.variables['asoil'][:][0][-1]   #unit = ug/kg-air
    
    '''
    2)write a new netCDF
    '''
    ft = Dataset('BEV_'+filelist[i],'w',format = 'NETCDF4')
     
    ntime = ft.createDimension('ntime',None)
    nlat = ft.createDimension('nlat',488)
    nlon = ft.createDimension('nlon',775)
    
    lat = ft.createVariable('lat','f4',('nlat','nlon'))
    lat.units =f.variables['lat'].units
    lat.description =f.variables['lat'].long_name
    
    lon = ft.createVariable('lon','f4',('nlat','nlon'))
    lon.units =f.variables['lon'].units
    lon.description =f.variables['lon'].long_name

    pm25 = ft.createVariable('pm25','f4',('ntime','nlat','nlon'))
    pm25.units ='ug m^-3'
    pm25.description ='pm2.5'
    
    # ps = ft.createVariable('ps','f4',('ntime','nlat','nlon'))
    # ps.units ='Pa'
    # ps.description ='pressure'
    
    # tmp = ft.createVariable('tmp','f4',('ntime','nlat','nlon'))
    # tmp.units = 'K'
    # tmp.description = 'Temperature'
    
    # uwnd = ft.createVariable('uwnd','f4',('ntime','nlat','nlon'))
    # uwnd.units =f.variables['ugrd'].units
    # uwnd.description ='wind velocity in u direction'
    
    # vwnd = ft.createVariable('vwnd','f4',('ntime','nlat','nlon'))
    # vwnd.units =f.variables['vgrd'].units
    # vwnd.description = 'wind velocity in v direction'
    
    vmro3 = ft.createVariable('o3','f4',('ntime','nlat','nlon'))
    vmro3.units ='ppbv'
    vmro3.description =f.variables['o3_ave'].long_name
    
    #vmrco = ft.createVariable('co','f4',('ntime','nlat','nlon'))
    #vmrco.units ='ppbv'
    #vmrco.description =f.variables['co'].long_name
    
    #vmrno = ft.createVariable('no','f4',('ntime','nlat','nlon'))
    #vmrno.units = 'ppbv'
    #vmrno.description =f.variables['no_ave'].long_name
    
    vmrno2 = ft.createVariable('no2','f4',('ntime','nlat','nlon'))
    vmrno2.units ='ppbv'
    vmrno2.description =f.variables['no2_ave'].long_name
   
    # mmrbc = ft.createVariable('mmrbc','f4',('ntime','nlon','nlat'))
    # mmrbc.units ='ug/kg-air'
    # mmrbc.description = 'total Black carbon concentration'
    
    # mmrso4 = ft.createVariable('mmrso4','f4',('ntime','nlon','nlat'))
    # mmrso4.units ='ug/kg-air'
    # mmrso4.description ='total sulfate concentration'

    # mmrnit = ft.createVariable('mmrnit','f4',('ntime','nlon','nlat'))
    # mmrnit.units ='ug/kg-air'
    # mmrnit.description ='total nitrate concentation'    

    # mmrnh4 = ft.createVariable('mmrnh4','f4',('ntime','nlon','nlat'))
    # mmrnh4.units ='ug/kg-air'
    # mmrnh4.description ='total ammonium concentration'
 
    # mmrsoil = ft.createVariable('mmrsoil','f4',('ntime','nlon','nlat'))
    # mmrsoil.units ='ug/kg-air'
    # mmrsoil.description ='total dust-soil concentration'
    
    '''
    3)input data
    '''
    lat[:,:]= lat_w
    lon[:,:]= lon_w
    pm25[0,:,:] = pm25_w
    # ps[0,:,:] = pressure_w
    # tmp[0,:,:] = temp_w
    # uwnd[0,:,:] = u_w
    # vwnd[0,:,:] = v_w
    vmro3[0,:,:] = vmro3_w
    #vmrco[0,:,:] = vmrco_w
    #vmrno[0,:,:]= vmrno_w
    vmrno2[0,:,:]= vmrno2_w
    # mmrbc[0,:,:] = mmrbc_w
    # mmrnit[0,:,:] = mmrnit_w
    # mmrso4[0,:,:] = mmrso4_w    
    # mmrnh4[0,:,:] = mmrnh4_w
    # mmrsoil[0,:,:] = mmrsoil_w
       
    ft.close()
    f.close()



