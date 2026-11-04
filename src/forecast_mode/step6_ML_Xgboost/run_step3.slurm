#!/bin/bash

#SBATCH --output=%j.out

#SBATCH --error=%j.err

#SBATCH --account=arl                                                                                                                                                                                                                
#SBATCH --qos=long                                                                                                                                                                                                                  
#SBATCH --partition=u1-compute

#SBATCH --nodes=1                                                                                                                                                                                                                    
#SBATCH --ntasks-per-node=1                                                                                                                                                                                                        
#SBATCH --cpus-per-task=16                                                                                                                                                                                                            
#SBATCH --time=00-23:00:00 

#SBATCH --mem=128G

#SBATCH --job-name="dafcom"


# Run Python
python step3_xgboost_training.py
