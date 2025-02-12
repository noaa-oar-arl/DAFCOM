import numpy as np
import pandas as pd
from xlwt import Workbook

dir_source = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/code/PM25_by_sample/2023/case4_forecast202308_train202306and202307/'
filename1 = 'CONUS_US_interpolated_2023_06_PM25.xls'
filename2 = 'CONUS_US_interpolated_2023_07_PM25.xls'
variable_name = 'PM25'
'''
1) read-in previous excels
'''
def GetData(dir_source_input,filename_input,variable_name_input):
    file_loc = dir_source_input
    obs_df = pd.read_excel(file_loc+filename_input)
    pd.set_option('display.max_columns',None)

    TIME = list(obs_df['Time_UTC'])
    LOC_NUMBER = list(obs_df['Location_number'])
    EPA_Variable = list(obs_df['EPA_OBS_'+'NO2'])

    V1_BLH = list(obs_df['V1_BLH'])
    V2_D2M = list(obs_df['V2_D2M'])
    V3_E = list(obs_df['V3_E'])
    V4_SP = list(obs_df['V4_SP'])
    V5_T2M = list(obs_df['V5_T2M'])
    V6_TP = list(obs_df['V6_TP'])
    V7_U10 = list(obs_df['V7_U10'])
    V8_V10 = list(obs_df['V8_V10'])
    V9_AOD = list(obs_df['V9_AOD'])
    
    V11_LAND_USE_COVER = list(obs_df['V11_LAND_USE_COVER'])
    V12_ELEVATION = list(obs_df['V12_ELEVATION'])
    V13_POPULATION = list(obs_df['V13_POPULATION'])
    V14_CTM_Variable = list(obs_df['V14_UFS_AQM_'+variable_name_input])

    V18_E_BC = list(obs_df['V18_E_BC'])
    V19_E_SO2 = list(obs_df['V19_E_SO2'])
    V20_E_NOX = list(obs_df['V20_E_NOX'])
    V21_E_VOC = list(obs_df['V21_E_VOC'])
    V22_E_NH3 = list(obs_df['V22_E_NH3'])
    V23_E_PM25 = list(obs_df['V23_E_PM25'])
    
    LAT = list(obs_df['LAT'])
    LON = list(obs_df['LON'])
    D1 = list(obs_df['D1'])
    D2 = list(obs_df['D2'])
    D3 = list(obs_df['D3'])
    D4 = list(obs_df['D4'])
    D5 = list(obs_df['D5'])
    JULIAN_DAY = list(obs_df['Julian_day'])


    return  TIME,LOC_NUMBER,EPA_Variable,V1_BLH,V2_D2M,V3_E,V4_SP,V5_T2M,V6_TP,V7_U10,V8_V10,V9_AOD,V11_LAND_USE_COVER,V12_ELEVATION,V13_POPULATION,V14_CTM_Variable,V18_E_BC,V19_E_SO2,V20_E_NOX,V21_E_VOC,V22_E_NH3,V23_E_PM25,LAT,LON,D1,D2,D3,D4,D5,JULIAN_DAY

TIME_1,LOC_NUMBER_1,EPA_Variable_1,V1_BLH_1,V2_D2M_1,V3_E_1,V4_SP_1,V5_T2M_1,V6_TP_1,V7_U10_1,V8_V10_1,V9_AOD_1,V11_LAND_USE_COVER_1,V12_ELEVATION_1,V13_POPULATION_1,V14_CTM_Variable_1,V18_E_BC_1,V19_E_SO2_1,V20_E_NOX_1,V21_E_VOC_1,V22_E_NH3_1,V23_E_PM25_1,LAT_1,LON_1,D1_1,D2_1,D3_1,D4_1,D5_1,JULIAN_DAY_1 = GetData(dir_source,filename1,variable_name)

TIME_2,LOC_NUMBER_2,EPA_Variable_2,V1_BLH_2,V2_D2M_2,V3_E_2,V4_SP_2,V5_T2M_2,V6_TP_2,V7_U10_2,V8_V10_2,V9_AOD_2,V11_LAND_USE_COVER_2,V12_ELEVATION_2,V13_POPULATION_2,V14_CTM_Variable_2,V18_E_BC_2,V19_E_SO2_2,V20_E_NOX_2,V21_E_VOC_2,V22_E_NH3_2,V23_E_PM25_2,LAT_2,LON_2,D1_2,D2_2,D3_2,D4_2,D5_2,JULIAN_DAY_2 = GetData(dir_source,filename2,variable_name)

'''
2) combine list
'''
TIME = TIME_1 + TIME_2
LOC_NUMBER =LOC_NUMBER_1 + LOC_NUMBER_2  
EPA_Variable = EPA_Variable_1 + EPA_Variable_2
V1_BLH= V1_BLH_1 + V1_BLH_2
V2_D2M= V2_D2M_1 + V2_D2M_2
V3_E = V3_E_1 + V3_E_2
V4_SP = V4_SP_1 + V4_SP_2
V5_T2M= V5_T2M_1 + V5_T2M_2
V6_TP = V6_TP_1 + V6_TP_2
V7_U10 = V7_U10_1 + V7_U10_2
V8_V10 = V8_V10_1 + V8_V10_2
V9_AOD = V9_AOD_1 + V9_AOD_2
V11_LAND_USE_COVER = V11_LAND_USE_COVER_1 + V11_LAND_USE_COVER_2
V12_ELEVATION = V12_ELEVATION_1 + V12_ELEVATION_2
V13_POPULATION = V13_POPULATION_1 + V13_POPULATION_2
V14_CTM_Variable = V14_CTM_Variable_1 + V14_CTM_Variable_2
V18_E_BC = V18_E_BC_1 + V18_E_BC_2
V19_E_SO2 = V19_E_SO2_1 + V19_E_SO2_2
V20_E_NOX =V20_E_NOX_1 + V20_E_NOX_2
V21_E_VOC =V21_E_VOC_1 + V21_E_VOC_2
V22_E_NH3 = V22_E_NH3_1 + V22_E_NH3_2
V23_E_PM25= V23_E_PM25_1 + V23_E_PM25_2
LAT = LAT_1 + LAT_2
LON = LON_1 + LON_2
D1 = D1_1+D1_2
D2 = D2_1+D2_2
D3 = D3_1+D3_2
D4 = D4_1+D4_2
D5 = D5_1+D5_2
JULIAN_DAY = JULIAN_DAY_1 + JULIAN_DAY_2

'''
3) write into a new excel
'''

wb = Workbook()
sheet1 = wb.add_sheet('EPA_PM25')

sheet1.write(0,0,'no')
sheet1.write(0,1,'Time_UTC')
sheet1.write(0,2,'Location_number')
sheet1.write(0,3,'EPA_OBS_PM25')
sheet1.write(0,5,'V1_BLH')
sheet1.write(0,6,'V2_D2M')
sheet1.write(0,7,'V3_E')
sheet1.write(0,8,'V4_SP')
sheet1.write(0,9,'V5_T2M')
sheet1.write(0,10,'V6_TP')
sheet1.write(0,11,'V7_U10')
sheet1.write(0,12,'V8_V10')
sheet1.write(0,13,'V9_AOD')
sheet1.write(0,15,'V11_LAND_USE_COVER')
sheet1.write(0,16,'V12_ELEVATION')
sheet1.write(0,17,'V13_POPULATION')
sheet1.write(0,18,'V14_UFS_AQM_PM25')
sheet1.write(0,22,'V18_E_BC')
sheet1.write(0,23,'V19_E_SO2')
sheet1.write(0,24,'V20_E_NOX')
sheet1.write(0,25,'V21_E_VOC')
sheet1.write(0,26,'V22_E_NH3')
sheet1.write(0,27,'V23_E_PM25')
sheet1.write(0,29,'LON')
sheet1.write(0,30,'LAT')
sheet1.write(0,31,'D1')
sheet1.write(0,32,'D2')
sheet1.write(0,33,'D3')
sheet1.write(0,34,'D4')
sheet1.write(0,35,'D5')
sheet1.write(0,36,'Julian_day')

for i in range(len(TIME)):
    sheet1.write(i+1,1,TIME[i])
    sheet1.write(i+1,2,LOC_NUMBER[i])
    sheet1.write(i+1,3,EPA_Variable[i])
    sheet1.write(i+1,5,V1_BLH[i])
    sheet1.write(i+1,6,V2_D2M[i])
    sheet1.write(i+1,7,V3_E[i])
    sheet1.write(i+1,8,V4_SP[i])
    sheet1.write(i+1,9,V5_T2M[i])
    sheet1.write(i+1,10,V6_TP[i])
    sheet1.write(i+1,11,V7_U10[i])
    sheet1.write(i+1,12,V8_V10[i])
    sheet1.write(i+1,13,V9_AOD[i])
    sheet1.write(i+1,15,V11_LAND_USE_COVER[i])
    sheet1.write(i+1,16,V12_ELEVATION[i])
    sheet1.write(i+1,17,V13_POPULATION[i])
    sheet1.write(i+1,18,V14_CTM_Variable[i])
    sheet1.write(i+1,22,V18_E_BC[i])
    sheet1.write(i+1,23,V19_E_SO2[i])
    sheet1.write(i+1,24,V20_E_NOX[i])
    sheet1.write(i+1,25,V21_E_VOC[i])
    sheet1.write(i+1,26,V22_E_NH3[i])
    sheet1.write(i+1,27,V23_E_PM25[i])
    sheet1.write(i+1,29,LON[i])
    sheet1.write(i+1,30,LAT[i])
    sheet1.write(i+1,31,D1[i])
    sheet1.write(i+1,32,D2[i])
    sheet1.write(i+1,33,D3[i])
    sheet1.write(i+1,34,D4[i])
    sheet1.write(i+1,35,D5[i])
    sheet1.write(i+1,36,JULIAN_DAY[i])


wb.save('CONUS_US_interpolated_2023_06and07_PM25.xls')

























