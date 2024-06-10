#!/bin/bash

#SBATCH --job-name=variant_all_soft_c012s_c1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=9216Mb
#SBATCH --time=06:00:00
#SBATCH --output="results/slurm_scripts_logs/output-%A_%a.out"
#SBATCH --error="results/slurm_scripts_logs/error-%A_%a.err"

cd $SLURM_SUBMIT_DIR
module load python/3.11.3-gcccore-12.3.0
source .venv/bin/activate

# Training
python main.py --action "train" --input "data_z" --num_classes "51" --epochs $EPOCHS \
--training_method "soft_c012s" --model_location "variant_all/51_classes_c1" \
--hits_c012s "hits_c1s" --features Variant-All

deactivate