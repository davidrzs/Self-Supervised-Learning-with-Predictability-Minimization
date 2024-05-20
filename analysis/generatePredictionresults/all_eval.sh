#!/bin/bash

# Base directory
BASE_DIR="imagenet100/"

# Find all subdirectories
for SUBDIR in $(find $BASE_DIR/* -type d)
do
    SUBDIR_ONLY=$(basename $SUBDIR)
    # Create a job script for this directory
    JOB_SCRIPT="slurm_job_$SUBDIR_ONLY.sh"

    # Write the job script
    cat > $JOB_SCRIPT << EOF
#!/bin/bash
#SBATCH --job-name=SMALL_EVAL
#SBATCH -n 1
#SBATCH --cpus-per-task=3
#SBATCH --time=100:00:00
#SBATCH --job-name="eval"
#SBATCH --mem-per-cpu=7000

module load gcc/8.2.0 python_gpu/3.10.4

# Run the python scripts
# python -m pipenv run python analysis/predict_all.py --model_type linear --folder '$SUBDIR/'
python -m pipenv run python analysis/predict_all.py --model_type xgb --folder '$SUBDIR/'
# python -m pipenv run python analysis/predict_all.py --model_type mlp --folder '$SUBDIR/'


# End the script with exit code 0
exit 0
EOF

    # Submit the job
    sbatch $JOB_SCRIPT
done