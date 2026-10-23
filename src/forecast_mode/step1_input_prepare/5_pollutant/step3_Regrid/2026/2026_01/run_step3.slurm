#!/bin/bash

#SBATCH --output=%j.out

#SBATCH --error=%j.err

#SBATCH --account=arl                                                                                                                                                                                                                
#SBATCH --qos=batch                                                                                                                                                                                                                  
#SBATCH --partition=u1-compute

#SBATCH --nodes=1                                                                                                                                                                                                                    
#SBATCH --ntasks-per-node=1                                                                                                                                                                                                        
#SBATCH --cpus-per-task=8                                                                                                                                                                                                            
#SBATCH --time=00-02:00:00 
#SBATCH --mem=256G
#SBATCH --job-name="dafcom_step1"


# Run Python
python step3_regrid_0p01_pm25.py
python step3_regrid_0p01_o3.py
python step3_regrid_0p01_no2.py
python step3_regrid_0p01_no.py
python step3_regrid_0p01_nox.py
