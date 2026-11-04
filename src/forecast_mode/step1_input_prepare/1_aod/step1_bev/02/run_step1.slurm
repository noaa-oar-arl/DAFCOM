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
#SBATCH --job-name="dafcom_aod"


# Run Python
python step1_extract_BEV.py
