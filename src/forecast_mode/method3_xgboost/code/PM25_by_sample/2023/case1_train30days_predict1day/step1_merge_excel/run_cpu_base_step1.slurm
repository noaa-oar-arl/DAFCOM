#!/bin/bash
## Job name
## partition=normal (CPU), =gpuq (GPU)
## GPU jobs must include 'qos', CPU do not
#SBATCH   --partition=contrib
#SBATCH   --qos=qtong
#SBATCH   --job-name=base 

## Standard ouput and error to file
## %u=userID, %x=jobName, %N=nodeID, %j=jobID,%A=arrayID, %a=arrayTaskID
#SBATCH   --output=%u-%x-%j.out        
#SBATCH   --error=%u-%x-%j.err      

## Wall time
#SBATCH   --time=3-00:00:00

## Number of nodes
## no MPI, use only 1 node
#SBATCH   --nodes=1

## Memory per CPU
#SBATCH   --cpus-per-task=4
#SBATCH   --mem-per-cpu=10GB

## Mail type
#SBATCH --mail-type=ALL

## Mail address
#SBATCH --mail-user=btang6@gmu.edu

## Load modules and activate my own environment
module load python
source /home/btang6/python_env_1st/AOD_fill/py-env/bin/activate  

## Run our job
python step1_merge_excel.py
