#!/bin/bash

#SBATCH --output=%j.out

#SBATCH --error=%j.err

#SBATCH --account=arl                                                                                                                                                                                                                
#SBATCH --qos=batch                                                                                                                                                                                                                  
#SBATCH --partition=u1-compute

#SBATCH --nodes=1                                                                                                                                                                                                                    
#SBATCH --ntasks-per-node=1                                                                                                                                                                                                        
#SBATCH --cpus-per-task=16                                                                                                                                                                                                            
#SBATCH --time=00-02:00:00 

#SBATCH --mem=256G

#SBATCH --job-name="dafcom_step3"


# Run Python
python step2_interpolate.py

