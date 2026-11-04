#!/bin/bash
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --account=arl
#SBATCH --qos=batch
#SBATCH --partition=u1-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=256G
#SBATCH --job-name="step5"

echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Requested memory: 256 GB"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "================================"

# Run Python and record resource usage
/usr/bin/time -v -o "${SLURM_JOB_ID}.usage" \
    python -u step2_LSTM_predict_bias.py

status=$?

echo ""
echo "========== RESOURCE USAGE =========="

# Show elapsed runtime and CPU usage
grep -E "Elapsed|Percent of CPU|Maximum resident set size" \
    "${SLURM_JOB_ID}.usage"

# Convert peak RAM from KB to GiB
awk -F: '/Maximum resident set size/ {
    printf "Peak memory: %.2f GiB\n", $2/1024/1024
}' "${SLURM_JOB_ID}.usage"

echo "End time: $(date)"
echo "Exit code: $status"
echo "===================================="

exit $status

