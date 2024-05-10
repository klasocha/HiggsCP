#!/bin/bash

#SBATCH --job-name=variant_1.1_data_preparation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=9216Mb
#SBATCH --time=01:00:00
#SBATCH --output="results/slurm_scripts_logs/output-%A_%a.out"
#SBATCH --error="results/slurm_scripts_logs/error-%A_%a.err"

cd $SLURM_SUBMIT_DIR
module load python/3.11.3-gcccore-12.3.0
source .venv/bin/activate

python main.py --action "download_and_preprocess" --input "data" \
--features "Variant-1.1" --num_classes "51" --hits_c012s "hits_c0s"

python main.py --action "download_and_preprocess" --input "data" \
--features "Variant-1.1" --num_classes "51" --hits_c012s "hits_c1s"

python main.py --action "download_and_preprocess" --input "data" \
--features "Variant-1.1" --num_classes "51" --hits_c012s "hits_c2s"

python main.py --action "download_and_preprocess" --input "data" \
--features "Variant-1.1" --num_classes "21" --hits_c012s "hits_c0s"

python main.py --action "download_and_preprocess" --input "data" \
--features "Variant-1.1" --num_classes "21" --hits_c012s "hits_c1s"

python main.py --action "download_and_preprocess" --input "data" \
--features "Variant-1.1" --num_classes "21" --hits_c012s "hits_c2s"

deactivate