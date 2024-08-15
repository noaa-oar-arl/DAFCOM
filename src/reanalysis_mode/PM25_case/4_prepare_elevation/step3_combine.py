#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:58:28 2022

@author: btang1
"""


from netCDF4 import Dataset
import numpy as np
from PIL import Image


'''
1) read in small domain
'''
def GetSmallDomain(input_pattern_index):
    dir_1 = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/data_CONUS_US/4_elevation/step2_regrid/'
    
    pattern1 = 'Elevation_0p01_regrid_'+input_pattern_index+'.nc'
    filename1 = dir_1 + pattern1
    f1 = Dataset(filename1,'r')   
    # v_list1 = list(f1.variables) 
    
    elevation1 = f1.variables['elevation'][:][0]
    lat1 = f1.variables['lat'][:]
    lon1 = f1.variables['lon'][:]
    
    return elevation1,lat1, lon1

elevation_1203,lat_1203,lon_1203 = GetSmallDomain('1203')
elevation_1204,lat_1204,lon_1204 = GetSmallDomain('1204')
elevation_1205,lat_1205,lon_1205 = GetSmallDomain('1205')
elevation_1206,lat_1206,lon_1206 = GetSmallDomain('1206')

elevation_1303,lat_1303,lon_1303 = GetSmallDomain('1303')
elevation_1304,lat_1304,lon_1304 = GetSmallDomain('1304')
elevation_1305,lat_1305,lon_1305 = GetSmallDomain('1305')
elevation_1306,lat_1306,lon_1306 = GetSmallDomain('1306')
elevation_1307,lat_1307,lon_1307 = GetSmallDomain('1307')

elevation_1403,lat_1403,lon_1403 = GetSmallDomain('1403')
elevation_1404,lat_1404,lon_1404 = GetSmallDomain('1404')
elevation_1405,lat_1405,lon_1405 = GetSmallDomain('1405')
elevation_1406,lat_1406,lon_1406 = GetSmallDomain('1406')
elevation_1407,lat_1407,lon_1407 = GetSmallDomain('1407')

elevation_1503,lat_1503,lon_1503 = GetSmallDomain('1503')
elevation_1504,lat_1504,lon_1504 = GetSmallDomain('1504')
elevation_1505,lat_1505,lon_1505 = GetSmallDomain('1505')
elevation_1506,lat_1506,lon_1506 = GetSmallDomain('1506')
elevation_1507,lat_1507,lon_1507 = GetSmallDomain('1507')

elevation_1603,lat_1603,lon_1603 = GetSmallDomain('1603')
elevation_1604,lat_1604,lon_1604 = GetSmallDomain('1604')
elevation_1605,lat_1605,lon_1605 = GetSmallDomain('1605')
elevation_1606,lat_1606,lon_1606 = GetSmallDomain('1606')
elevation_1607,lat_1607,lon_1607 = GetSmallDomain('1607')

elevation_1703,lat_1703,lon_1703 = GetSmallDomain('1703')
elevation_1704,lat_1704,lon_1704 = GetSmallDomain('1704')
elevation_1705,lat_1705,lon_1705 = GetSmallDomain('1705')
elevation_1706,lat_1706,lon_1706 = GetSmallDomain('1706')
elevation_1707,lat_1707,lon_1707 = GetSmallDomain('1707')

elevation_1803,lat_1803,lon_1803 = GetSmallDomain('1803')
elevation_1804,lat_1804,lon_1804 = GetSmallDomain('1804')
elevation_1805,lat_1805,lon_1805 = GetSmallDomain('1805')
elevation_1806,lat_1806,lon_1806 = GetSmallDomain('1806')
elevation_1807,lat_1807,lon_1807 = GetSmallDomain('1807')

elevation_1903,lat_1903,lon_1903 = GetSmallDomain('1903')
elevation_1904,lat_1904,lon_1904 = GetSmallDomain('1904')
elevation_1905,lat_1905,lon_1905 = GetSmallDomain('1905')
elevation_1906,lat_1906,lon_1906 = GetSmallDomain('1906')
elevation_1907,lat_1907,lon_1907 = GetSmallDomain('1907')

elevation_2003,lat_2003,lon_2003 = GetSmallDomain('2003')
elevation_2004,lat_2004,lon_2004 = GetSmallDomain('2004')
elevation_2005,lat_2005,lon_2005 = GetSmallDomain('2005')
elevation_2006,lat_2006,lon_2006 = GetSmallDomain('2006')
elevation_2007,lat_2007,lon_2007 = GetSmallDomain('2007')

elevation_2103,lat_2103,lon_2103 = GetSmallDomain('2103')
elevation_2104,lat_2104,lon_2104 = GetSmallDomain('2104')
elevation_2105,lat_2105,lon_2105 = GetSmallDomain('2105')
elevation_2106,lat_2106,lon_2106 = GetSmallDomain('2106')
elevation_2107,lat_2107,lon_2107 = GetSmallDomain('2107')

elevation_2203,lat_2203,lon_2203 = GetSmallDomain('2203')
elevation_2204,lat_2204,lon_2204 = GetSmallDomain('2204')
elevation_2205,lat_2205,lon_2205 = GetSmallDomain('2205')

elevation_2303,lat_2303,lon_2303 = GetSmallDomain('2303')
elevation_2304,lat_2304,lon_2304 = GetSmallDomain('2304')


'''
1-2) create small domain 1207, which is all ocaen ==0
'''
lat_1207 = lat_1307
lon_1207 = lon_1206
elevation_1207 = np.zeros((500,500))

lat_2206 = lat_2106
lon_2206 = lon_2205
elevation_2206 = np.zeros((500,500))

lat_2207 = lat_2107
lon_2207  = lon_2205
elevation_2207 = np.zeros((500,500))

lat_2305 = lat_2105
lon_2305 = lon_2304
elevation_2305 = np.zeros((500,500))

lat_2306 = lat_2106
lon_2306 = lon_2304
elevation_2306 = np.zeros((500,500))

lat_2307 = lat_2107
lon_2307 = lon_2304
elevation_2307 = np.zeros((500,500))

'''
2) combine columns
'''
elevation_combine_col12 = np.concatenate((elevation_1203,elevation_1204,elevation_1205,
                                          elevation_1206,elevation_1207), axis = 0)

elevation_combine_col13 = np.concatenate((elevation_1303,elevation_1304,elevation_1305,
                                          elevation_1306,elevation_1307), axis = 0)

elevation_combine_col14 = np.concatenate((elevation_1403,elevation_1404,elevation_1405,
                                          elevation_1406,elevation_1407), axis = 0)

elevation_combine_col15 = np.concatenate((elevation_1503,elevation_1504,elevation_1505,
                                          elevation_1506,elevation_1507), axis = 0)

elevation_combine_col16 = np.concatenate((elevation_1603,elevation_1604,elevation_1605,
                                          elevation_1606,elevation_1607), axis = 0)

elevation_combine_col17 = np.concatenate((elevation_1703,elevation_1704,elevation_1705,
                                          elevation_1706,elevation_1707), axis = 0)

elevation_combine_col18 = np.concatenate((elevation_1803,elevation_1804,elevation_1805,
                                          elevation_1806,elevation_1807), axis = 0)

elevation_combine_col19 = np.concatenate((elevation_1903,elevation_1904,elevation_1905,
                                          elevation_1906,elevation_1907), axis = 0)

elevation_combine_col20 = np.concatenate((elevation_2003,elevation_2004,elevation_2005,
                                          elevation_2006,elevation_2007), axis = 0)

elevation_combine_col21 = np.concatenate((elevation_2103,elevation_2104,elevation_2105,
                                          elevation_2106,elevation_2107), axis = 0)

elevation_combine_col22 = np.concatenate((elevation_2203,elevation_2204,elevation_2205,
                                          elevation_2206,elevation_2207), axis = 0)

elevation_combine_col23 = np.concatenate((elevation_2303,elevation_2304,elevation_2305,
                                          elevation_2306,elevation_2307), axis = 0)

'''
3) combine all columns to big domain
'''
elevation_combine_westUS = np.concatenate((elevation_combine_col12,elevation_combine_col13,elevation_combine_col14,
                                           elevation_combine_col15,elevation_combine_col16,elevation_combine_col17,
                                           elevation_combine_col18,elevation_combine_col19,elevation_combine_col20,
                                           elevation_combine_col21,elevation_combine_col22,elevation_combine_col23
                                           ), axis = 1)

elevation_combine_westUS_nlayer = elevation_combine_westUS .reshape((1,elevation_combine_westUS .shape[0],elevation_combine_westUS.shape[1]))

lat_WestUS = [50- 0.01*X for X in range(2500)]
lon_WestUS = [-125+ 0.01*X for X in range(6000)]


'''
4) write into nc file
'''

ft = Dataset(('Elevation_0p01_regird'+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',2500)
nlon = ft.createDimension('longitude',6000)
nlayer = ft.createDimension('time',1)


lat_new_nc = ft.createVariable('latitude','f8',('latitude'))
lat_new_nc.units =''
lat_new_nc.long_name ='latitude'

lon_new_nc = ft.createVariable('longitude','f8',('longitude'))
lon_new_nc.units =''
lon_new_nc.long_name ='longitude'

variable_new = ft.createVariable('Elevation','f8',('time','latitude','longitude'))  
variable_new .units =''                                               
variable_new .description =''

lat_new_nc[:] = np.array(lat_WestUS )
lon_new_nc[:] = np.array(lon_WestUS)
variable_new[:,:,:] = np.array(elevation_combine_westUS_nlayer)

ft.close()






