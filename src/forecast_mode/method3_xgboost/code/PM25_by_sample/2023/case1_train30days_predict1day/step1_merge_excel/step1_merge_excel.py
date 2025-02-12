import numpy as np
import pandas as pd
from xlwt import Workbook

dir_source = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method2_30daysTrain_1dayPredict/code/PM25_by_sample/2023/case1_train30days_predict1day/'
filename1  = 'CONUS_US_interpolated_2023_07_PM25.xls'
filename2  = 'CONUS_US_interpolated_2023_08_PM25.xls'
variable_name = 'PM25'
start_date_list = [182+x for x in range(33)]  #06/01 is 152,07/02 is 183
end_date_list   = [211+x for x in range(33)]  #06/30 is 181,07/31 is 212
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

for iii in range(len(start_date_list)):
    start_date = start_date_list[iii]
    end_date   = end_date_list[iii]

    TIME_final = []
    LOC_NUMBER_final = []
    EPA_Variable_final = []
    V1_BLH_final = []
    V2_D2M_final = []
    V3_E_final = []
    V4_SP_final = []
    V5_T2M_final = []
    V6_TP_final = []
    V7_U10_final = []
    V8_V10_final = []
    V9_AOD_final = []
    V11_LAND_USE_COVER_final = []
    V12_ELEVATION_final = []
    V13_POPULATION_final = []
    V14_CTM_Variable_final = []
    V18_E_BC_final = []
    V19_E_SO2_final = []
    V20_E_NOX_final = []
    V21_E_VOC_final = []
    V22_E_NH3_final = []
    V23_E_PM25_final = []
    LAT_final = []
    LON_final = []
    D1_final = []
    D2_final = []
    D3_final = []
    D4_final = []
    D5_final = []
    JULIAN_DAY_final = []

    for i in range(len(JULIAN_DAY)):
        if JULIAN_DAY[i] >=start_date and JULIAN_DAY[i] <= end_date:
            TIME_final.append(TIME[i])
            LOC_NUMBER_final.append(LOC_NUMBER[i])
            EPA_Variable_final.append(EPA_Variable[i])
            V1_BLH_final.append(V1_BLH[i])
            V2_D2M_final.append(V2_D2M[i])
            V3_E_final.append(V3_E[i])
            V4_SP_final.append(V4_SP[i])
            V5_T2M_final.append(V5_T2M[i])
            V6_TP_final.append(V6_TP[i])
            V7_U10_final.append(V7_U10[i])
            V8_V10_final.append(V8_V10[i])
            V9_AOD_final.append(V9_AOD[i])
            V11_LAND_USE_COVER_final.append(V11_LAND_USE_COVER[i])
            V12_ELEVATION_final.append(V12_ELEVATION[i])
            V13_POPULATION_final.append(V13_POPULATION[i])
            V14_CTM_Variable_final.append(V14_CTM_Variable[i])
            V18_E_BC_final.append(V18_E_BC[i])
            V19_E_SO2_final.append(V19_E_SO2[i])
            V20_E_NOX_final.append(V20_E_NOX[i])
            V21_E_VOC_final.append(V21_E_VOC[i])
            V22_E_NH3_final.append(V22_E_NH3[i])
            V23_E_PM25_final.append(V23_E_PM25[i])
            LAT_final.append(LAT[i])
            LON_final.append(LON[i])
            D1_final.append(D1[i])
            D2_final.append(D2[i])
            D3_final.append(D3[i])
            D4_final.append(D4[i])
            D5_final.append(D5[i])
            JULIAN_DAY_final.append(JULIAN_DAY[i])

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

    for i in range(len(TIME_final)):
        sheet1.write(i+1,1,TIME_final[i])
        sheet1.write(i+1,2,LOC_NUMBER_final[i])
        sheet1.write(i+1,3,EPA_Variable_final[i])
        sheet1.write(i+1,5,V1_BLH_final[i])
        sheet1.write(i+1,6,V2_D2M_final[i])
        sheet1.write(i+1,7,V3_E_final[i])
        sheet1.write(i+1,8,V4_SP_final[i])
        sheet1.write(i+1,9,V5_T2M_final[i])
        sheet1.write(i+1,10,V6_TP_final[i])
        sheet1.write(i+1,11,V7_U10_final[i])
        sheet1.write(i+1,12,V8_V10_final[i])
        sheet1.write(i+1,13,V9_AOD_final[i])
        sheet1.write(i+1,15,V11_LAND_USE_COVER_final[i])
        sheet1.write(i+1,16,V12_ELEVATION_final[i])
        sheet1.write(i+1,17,V13_POPULATION_final[i])
        sheet1.write(i+1,18,V14_CTM_Variable_final[i])
        sheet1.write(i+1,22,V18_E_BC_final[i])
        sheet1.write(i+1,23,V19_E_SO2_final[i])
        sheet1.write(i+1,24,V20_E_NOX_final[i])
        sheet1.write(i+1,25,V21_E_VOC_final[i])
        sheet1.write(i+1,26,V22_E_NH3_final[i])
        sheet1.write(i+1,27,V23_E_PM25_final[i])
        sheet1.write(i+1,29,LON_final[i])
        sheet1.write(i+1,30,LAT_final[i])
        sheet1.write(i+1,31,D1_final[i])
        sheet1.write(i+1,32,D2_final[i])
        sheet1.write(i+1,33,D3_final[i])
        sheet1.write(i+1,34,D4_final[i])
        sheet1.write(i+1,35,D5_final[i])
        sheet1.write(i+1,36,JULIAN_DAY_final[i])


    wb.save('CONUS_US_interpolated_2023_'+str(start_date)+'_'+str(end_date)+'_PM25.xls')

























