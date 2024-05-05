#!/bin/bash

#SBATCH --job-name=variant_1.1_soft_argmaxs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3850M
#SBATCH --time=02:00:00
#SBATCH --output="results/slurm_scripts_logs/output-%A_%a.out"
#SBATCH --error="results/slurm_scripts_logs/error-%A_%a.err"

cd $SLURM_SUBMIT_DIR
module load python/3.11.3-gcccore-12.3.0
source .venv/bin/activate

python main.py --action "train" --input "data" --num_classes "21" --epochs "120" --training_method "soft_argmaxs" --model_location "21_classes_variant_1.1"

python main.py --action "predict_test" --input "data" --num_classes "21" --training_method "soft_argmaxs" --model_location "21_classes_variant_1.1" --use_filtered_data
python main.py --action "plot" --input "results/soft_argmaxs/21_classes/predictions" --output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_3" --features Variant-1.1 --dataset "test" --num_classes "21" 
python main.py --action "plot" --input "results/soft_argmaxs/21_classes/predictions" --output "plots/figures" --format "pdf" --option "RESULTS_ANALYSIS_3" --features Variant-1.1 --dataset "test" --num_classes "21" 
python main.py --action "plot" --input "results/soft_argmaxs/21_classes/predictions" --output "plots/figures" --format "eps" --option "RESULTS_ANALYSIS_3" --features Variant-1.1 --dataset "test" --num_classes "21" 

python main.py --action "predict_train_and_valid" --input "data" --num_classes "21" --training_method "soft_argmaxs" --model_location "21_classes_variant_1.1" --use_filtered_data
python main.py --action "plot" --input "results/soft_argmaxs/21_classes/predictions" --output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_3" --features Variant-1.1 --dataset "valid" --num_classes "21" 
python main.py --action "plot" --input "results/soft_argmaxs/21_classes/predictions" --output "plots/figures" --format "pdf" --option "RESULTS_ANALYSIS_3" --features Variant-1.1 --dataset "valid" --num_classes "21" 
python main.py --action "plot" --input "results/soft_argmaxs/21_classes/predictions" --output "plots/figures" --format "eps" --option "RESULTS_ANALYSIS_3" --features Variant-1.1 --dataset "valid" --num_classes "21" 

deactivate