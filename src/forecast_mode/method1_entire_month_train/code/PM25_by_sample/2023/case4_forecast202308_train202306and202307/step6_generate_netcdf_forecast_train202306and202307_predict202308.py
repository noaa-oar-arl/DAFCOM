#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:46:06 2022

@author: btang1
"""
import numpy as np
from netCDF4 import Dataset
import joblib

'''
1) load saved random forest algorithm
'''
dir_ = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/code/PM25_by_sample/2023/case4_forecast202308_train202306and202307/'
loaded_rf = joblib.load(dir_+'rf_train_2023_06and07.joblib')
'''
2) prepare input data 14
'''
from step1_Daily_Avr_InputMatrix import BLH, D2M, E, SP, T2M, TP, U10, V10, LAT, LON
from step1_Daily_Avr_InputMatrix import AOD, Land_Use_Cover, Elevation, Population
from step1_Daily_Avr_InputMatrix import UFS_AQM_PM25_daily, E_BC_daily, E_SO2_daily, E_NOX_daily, E_VOC_daily, E_NH3_daily, E_PM25_daily
from step1_Daily_Avr_InputMatrix import Lon_matrix_daily ,Lat_matrix_daily, D1_matrix_daily, D2_matrix_daily,D3_matrix_daily, D4_matrix_daily, D5_matrix_daily, day_matrix_daily

print(np.shape(BLH))

V1 = BLH
V2 = D2M
V3 = E
V4 = SP
V5 = T2M
V6 = TP
V7 = U10
V8 = V10
V9 = AOD
#V10 = NDVI
V11 = Land_Use_Cover
V12 = Elevation
V13 = Population
V14 = UFS_AQM_PM25_daily
# V15 = CAMS_o3_daily
# V16 = CAMS_no_daily
#V17 = CAMS_no2_daily
V18 = E_BC_daily
V19 = E_SO2_daily
V20 = E_NOX_daily
V21 = E_VOC_daily
V22 = E_NH3_daily
V23 = E_PM25_daily
#V24 = TROP_NO2

V25 = Lat_matrix_daily
V26 = Lon_matrix_daily
V27 = D1_matrix_daily
V28 = D2_matrix_daily
V29 = D3_matrix_daily
V30 = D4_matrix_daily
V31 = D5_matrix_daily
V32 = day_matrix_daily


'''
2) make daily prediction then average to monthly
'''
def GetPredDay(day_index):  #this is per day prediction
    V1_long = np.reshape(V1[day_index],(14400000,1))
    V2_long = np.reshape(V2[day_index],(14400000,1))
    V3_long = np.reshape(V3[day_index],(14400000,1))
    V4_long = np.reshape(V4[day_index],(14400000,1))
    V5_long = np.reshape(V5[day_index],(14400000,1))
    V6_long = np.reshape(V6[day_index],(14400000,1))
    V7_long = np.reshape(V7[day_index],(14400000,1))
    V8_long = np.reshape(V8[day_index],(14400000,1))
    V9_long = np.reshape(V9[day_index],(14400000,1))
    #V10_long = np.reshape(V10[day_index],(14400000,1))
    V11_long = np.reshape(V11[day_index],(14400000,1))
    V12_long = np.reshape(V12[day_index],(14400000,1))
    V13_long = np.reshape(V13[day_index],(14400000,1))
    V14_long = np.reshape(V14[day_index],(14400000,1))
    
    V18_long = np.reshape(V18[day_index],(14400000,1))
    V19_long = np.reshape(V19[day_index],(14400000,1))
    V20_long = np.reshape(V20[day_index],(14400000,1))
    V21_long = np.reshape(V21[day_index],(14400000,1))
    V22_long = np.reshape(V22[day_index],(14400000,1))
    V23_long = np.reshape(V23[day_index],(14400000,1))   
    #V24_long = np.reshape(V24[day_index],(14400000,1))
    
    V25_long = np.reshape(V25[day_index],(14400000,1))
    V26_long = np.reshape(V26[day_index],(14400000,1))
    V27_long = np.reshape(V27[day_index],(14400000,1))
    V28_long = np.reshape(V28[day_index],(14400000,1))
    V29_long = np.reshape(V29[day_index],(14400000,1))
    V30_long = np.reshape(V30[day_index],(14400000,1))
    V31_long = np.reshape(V31[day_index],(14400000,1))
    V32_long = np.reshape(V32[day_index],(14400000,1))
    
    input_test = np.reshape(np.transpose([V1_long,V2_long,V3_long,V4_long,V5_long,V6_long,V7_long,V8_long,V9_long,V11_long,V12_long,V13_long,V14_long,V18_long,V19_long,V20_long,V21_long,V22_long,V23_long,V25_long,V26_long,V27_long,V28_long,V29_long,V30_long,V31_long,V32_long]),(14400000,27))
    
    
    Y_hat_long = loaded_rf.predict(input_test)
    Y_hat = np.reshape(Y_hat_long,(2400,6000)) 

    return Y_hat          


# Y_hat_month = GetPredDay(0)
# for i in range(1, len(BLH)): #this is 30 days
#     print(i)
#     Y_hat_day = GetPredDay(i)
#     Y_hat_month += Y_hat_day
        
# Y_hat_month = Y_hat_month/30

'''
supplemental save in netcdf file
'''
Y_hat_month_daily = []
for i in range(31):
    print(i)
    Y_hat_day = GetPredDay(i)
    Y_hat_month_daily.append(Y_hat_day)
    
lat_WestUS = [49- 0.01*X for X in range(2400)]
lon_WestUS = [-125+ 0.01*X for X in range(6000)]    
    
ft = Dataset(('forecast202308_train202306and202307_pm25_0p01'+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',2400)
nlon = ft.createDimension('longitude',6000)
nlayer = ft.createDimension('time',31)


lat_new_nc = ft.createVariable('latitude','f8',('latitude'))
lat_new_nc.units =''
lat_new_nc.long_name ='latitude'

lon_new_nc = ft.createVariable('longitude','f8',('longitude'))
lon_new_nc.units =''
lon_new_nc.long_name ='longitude'

variable_new = ft.createVariable('pm25','f8',('time','latitude','longitude'))  
variable_new .units ='ug/m^3'                                               
variable_new .description =''

lat_new_nc[:] = np.array(lat_WestUS )
lon_new_nc[:] = np.array(lon_WestUS)
variable_new[:,:,:] = np.array(Y_hat_month_daily )

ft.close()


            
'''
4) plot result into a map
'''
'''
4-1) CONUS domain
'''
# dir_1 = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/NO2_Case/data_CONUS_US/0_meteo/step2_0p01/'
# pattern_0 = 'meteo_blh_0p01.nc'
# filename_0 = dir_1 + pattern_0
# f0 = Dataset(filename_0,'r')
# LAT = f0.variables['lat'][:] #2500
# LON = f0.variables['lon'][:] #3500

# dir_2 = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/NO2_Case/code_CONUS_US/no2_splitbysample_Mcmaq_Egmu_tropno2/'
# pattern_1 = 'CONUS_US_surfaceNO2_0p01_regird.nc'
# filename_1 = dir_2 + pattern_1
# f1 = Dataset(filename_1,'r')
# # print(list(f1.variables))
# Y_hat_month_daily = f1.variables['Surface no2'][:] #(30,2500,3500)


# Y_hat_month = np.mean(Y_hat_month_daily ,axis = 0)

# # #plot monthly averge map 
# m = Basemap(projection='cyl', resolution='l',llcrnrlat=25, urcrnrlat = 49,llcrnrlon=-125, urcrnrlon = -65) 
# fig = plt.subplots(1,1,figsize = (15,10)) 
# m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 25,cmap = 'jet') 

# m.drawcoastlines(linewidth = 2)
# m.drawcountries(linewidth = 2)
# m.drawstates(linewidth = 2)

# #add shape file
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')       
# plt.savefig('ML_NO2_CONUS.png', dpi = 600)


'''
4-2) California
'''

# m = Basemap(projection='cyl', resolution='l',llcrnrlat=31, urcrnrlat = 43,llcrnrlon=-125, urcrnrlon = -114) 
# fig = plt.subplots(1,1,figsize = (15,10)) 
# m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 25,cmap = 'jet') 

# cb = m.colorbar()
# cb.ax.tick_params(labelsize=30)
# cb.set_label('ppbv',fontsize = 30) 

# # m.drawparallels(np.arange(25, 50, 10), labels=[1, 0, 0, 0],fontsize = 30)
# # m.drawmeridians(np.arange(-125, -90, 10), labels=[0, 0, 0, 1],fontsize = 30)

# m.drawcoastlines(linewidth = 2)
# m.drawcountries(linewidth = 2)
# m.drawstates(linewidth = 2)
                                       
# #add shape file
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')
# # plt.title('1km*1km resolution Monthly average $PM_{2.5}$ concentraion \n in May-2021 \nmethod4 CAMS input,GEMS AOD, add spatial&time input')
# plt.savefig('ML_NO2_California.png', dpi = 600)




'''
4-3) LA domain
'''


# m = Basemap(projection='cyl', resolution='l',llcrnrlat=33.7, urcrnrlat = 34.3,llcrnrlon=-118.7, urcrnrlon = -117.7) 
# fig = plt.subplots(1,1,figsize = (12,12)) 
# m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 35,cmap = 'jet') 

# cb = m.colorbar()
# cb.ax.tick_params(labelsize=30)
# cb.set_label('ppbv',fontsize = 30)   
                                       

# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/10_LA_road/'
# m.readshapefile(dir_shapefile+"tl_2018_06037_roads",'la_roads',linewidth = 0.3)

# # plt.title('1km*1km resolution Monthly average $PM_{2.5}$ concentraion \n in May-2021 \nmethod4 CAMS input, GEMS AOD, add spatial&time input')
# plt.savefig('ML_no2_LA.png', dpi = 300)



'''
4-4) Denver domain
'''


# m = Basemap(projection='cyl', resolution='l',llcrnrlat=39.65, urcrnrlat = 39.78,llcrnrlon=-105.05, urcrnrlon = -104.9) 
# fig = plt.subplots(1,1,figsize = (12,12)) 
# m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 15,cmap = 'jet') 

# cb = m.colorbar()
# cb.ax.tick_params(labelsize=30)
# cb.set_label('\u03bcg/$m^3$',fontsize = 30)   
                                       

# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/7_Denver_road/'
# m.readshapefile(dir_shapefile+"tl_2019_08031_roads",'denver_roads',linewidth = 0.3)

# # plt.title('1km*1km resolution Monthly average $PM_{2.5}$ concentraion \n in May-2021 \nmethod4 CAMS input, GEMS AOD, add spatial&time input')
# plt.savefig('ML_pm25_Denver.png', dpi = 300)

























