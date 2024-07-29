#!/bin/bash

#SBATCH --job-name=variant_1.0_data_preparation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=9216Mb
#SBATCH --time=02:00:00
#SBATCH --output="results/slurm_scripts_logs/output-%A_%a.out"
#SBATCH --error="results/slurm_scripts_logs/error-%A_%a.err"

cd $SLURM_SUBMIT_DIR
module load python/3.11.3-gcccore-12.3.0
source .venv/bin/activate

# Preparing data
mkdir data_z
scp data/unwt_multiclass_51.npy data_z/
python main.py --action "download_and_prepare_original" --input "data_z" --exp "Z"

deactivate