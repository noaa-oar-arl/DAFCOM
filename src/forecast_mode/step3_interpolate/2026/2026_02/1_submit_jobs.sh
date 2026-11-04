year=2026
month=02  

for day in {01..28}; do
        dir="${year}_${month}${day}"
        cd $dir
        sbatch run_step2.slurm
        cd ../
done

