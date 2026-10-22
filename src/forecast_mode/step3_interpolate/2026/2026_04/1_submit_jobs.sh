year=2026
month=04  

for day in {01..30}; do
        dir="${year}_${month}${day}"
        cd $dir
        sbatch run_step2.slurm
        cd ../
done

