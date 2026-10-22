year=2026
month=01  

for day in {01..31}; do
        dir="${year}_${month}${day}"
        cd $dir
        sbatch run_step2.slurm
        cd ../
done

