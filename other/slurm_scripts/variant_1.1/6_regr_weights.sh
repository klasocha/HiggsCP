#!/bin/bash

#SBATCH --job-name=variant_1.1_regr_weights
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3850M
#SBATCH --time=02:30:00
#SBATCH --output="results/slurm_scripts_logs/output-%A_%a.out"
#SBATCH --error="results/slurm_scripts_logs/error-%A_%a.err"

cd $SLURM_SUBMIT_DIR
module load python/3.11.3-gcccore-12.3.0
source .venv/bin/activate
    	
python main.py --action "train" --input "data" --num_classes "51" --epochs "120" --training_method "regr_weights" --model_location "51_classes_variant_1.1"
python main.py --action "predict_test" --input "data" --num_classes "51" --training_method "regr_weights" --model_location "51_classes_variant_1.1" --use_filtered_data

python main.py --action "plot" --input "results/regr_weights/51_classes_variant_1.1/predictions" --output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_1" --num_classes "51" --training_method "regr_weights" --features Variant-1.1 --dataset "test" --use_filtered_data
python main.py --action "plot" --input "results/regr_weights/51_classes_variant_1.1/predictions" --output "plots/figures" --format "pdf" --option "RESULTS_ANALYSIS_1" --num_classes "51" --training_method "regr_weights" --features Variant-1.1 --dataset "test" --use_filtered_data
python main.py --action "plot" --input "results/regr_weights/51_classes_variant_1.1/predictions" --output "plots/figures" --format "eps" --option "RESULTS_ANALYSIS_1" --num_classes "51" --training_method "regr_weights" --features Variant-1.1 --dataset "test" --use_filtered_data

deactivate