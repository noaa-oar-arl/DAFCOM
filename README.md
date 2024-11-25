# NOAA's DynAmic Forecasting of Air Composition using Optimized Machine Learning

description


### Major Goals 

* Goal 1: build 1st version workflow  
  - [ ] step 1  : process all inputs to be daily, 1 by 1km for June, July, Aug, 2023
  - [ ] step 2  : build initial forcast & downscaling Machine learning model with 4 cases: 1) train 2023-06, predict 2023-07; 2) train 2023-07, predict 2023-08; 3) train 2023-06, predict 2023-08; 4) train 2023-06&07, predict 2023-08
* Goal 2: build 2nd version workflow
  - [ ] step 1 : process all inputs for training with nearest 30 days, and predict only 1 day. repeat for predicting 2023-07-01 to 2023-08-31.
  - [ ] step 2 : analyze ML model preformance, and compared to Goal 1.
  - [ ] step 3 : now we use 30 days to train, update to determine the optimal time-scale/length of training.
### How to Run

currently, all code are located and stored in HOPPER HPC.
Located in: /groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/code/
intermediate data are located in: /groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/data/
all code are python, we have saved slurm file for runnig each python steps.
to run slurm and python, an enviroment 'DAFCOM' has been built on HOPPER HPC, and all requird libary has been installed in 'DAFCOM' enviroment. slurm will call 'DAFCOM' to run


### Time line of Funding

Timeline or finish date at the miminum
