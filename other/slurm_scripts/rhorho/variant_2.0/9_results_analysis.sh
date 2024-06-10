#!/bin/bash

#SBATCH --job-name=variant_2.0_results_analysis
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

# Plots (soft_weights)
python main.py --action "plot" --input "results/soft_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "soft_weights" --features Variant-2.0 --dataset "test" \
--use_filtered_data
python main.py --action "plot" --input "results/soft_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "soft_weights" --features Variant-2.0 --dataset "valid" \
--use_filtered_data
python main.py --action "plot" --input "results/soft_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "soft_weights" --features Variant-2.0 --dataset "train" \
--use_filtered_data

python main.py --action "plot" --input "results/soft_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "soft_weights" --features Variant-2.0 --dataset "test"
python main.py --action "plot" --input "results/soft_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "soft_weights" --features Variant-2.0 --dataset "valid"
python main.py --action "plot" --input "results/soft_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "soft_weights" --features Variant-2.0 --dataset "train"


# Plots (soft_c012s)
python main.py  --action "plot" --input "results/soft_c012s/variant_2.0/51_classes_c" \
--output "plots/figures" --option "RESULTS_ANALYSIS_2" --num_classes "51" \
--training_method "soft_c012s" --features Variant-2.0 --dataset "test" \
--use_filtered_data
python main.py --action "plot" --input "results/soft_c012s/variant_2.0/51_classes_c" \
--output "plots/figures" --option "RESULTS_ANALYSIS_2" --num_classes "51" \
--training_method "soft_c012s" --features Variant-2.0 --dataset "valid" \
--use_filtered_data
python main.py --action "plot" --input "results/soft_c012s/variant_2.0/51_classes_c" \
--output "plots/figures" --option "RESULTS_ANALYSIS_2" --num_classes "51" \
--training_method "soft_c012s" --features Variant-2.0 --dataset "train" \
--use_filtered_data

python main.py  --action "plot" --input "results/soft_c012s/variant_2.0/51_classes_c" \
--output "plots/figures" --option "RESULTS_ANALYSIS_2" --num_classes "51" \
--training_method "soft_c012s" --features Variant-2.0 --dataset "test" 
python main.py --action "plot" --input "results/soft_c012s/variant_2.0/51_classes_c" \
--output "plots/figures" --option "RESULTS_ANALYSIS_2" --num_classes "51" \
--training_method "soft_c012s" --features Variant-2.0 --dataset "valid" 
python main.py --action "plot" --input "results/soft_c012s/variant_2.0/51_classes_c" \
--output "plots/figures" --option "RESULTS_ANALYSIS_2" --num_classes "51" \
--training_method "soft_c012s" --features Variant-2.0 --dataset "train" 


# Plots (soft_argmaxs)
python main.py --action "plot" --input "results/soft_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_3" --features Variant-2.0 \
--training_method "soft_argmaxs" --dataset "test" --num_classes "51" --use_filtered_data
python main.py --action "plot" --input "results/soft_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_3" --features Variant-2.0 \
--training_method "soft_argmaxs" --dataset "valid" --num_classes "51" --use_filtered_data
python main.py --action "plot" --input "results/soft_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_3" --features Variant-2.0 \
--training_method "soft_argmaxs" --dataset "train" --num_classes "51" --use_filtered_data

python main.py --action "plot" --input "results/soft_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_3" --features Variant-2.0 \
--training_method "soft_argmaxs" --dataset "test" --num_classes "51"
python main.py --action "plot" --input "results/soft_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_3" --features Variant-2.0 \
--training_method "soft_argmaxs" --dataset "valid" --num_classes "51"
python main.py --action "plot" --input "results/soft_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_3" --features Variant-2.0 \
--training_method "soft_argmaxs" --dataset "train" --num_classes "51"


# Plots (regr_weights)
python main.py --action "plot" --input "results/regr_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "regr_weights" --features Variant-2.0 --dataset "test" --use_filtered_data
python main.py --action "plot" --input "results/regr_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "regr_weights" --features Variant-2.0 --dataset "train" --use_filtered_data
python main.py --action "plot" --input "results/regr_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "regr_weights" --features Variant-2.0 --dataset "valid" --use_filtered_data

python main.py --action "plot" --input "results/regr_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "regr_weights" --features Variant-2.0 --dataset "test"
python main.py --action "plot" --input "results/regr_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "regr_weights" --features Variant-2.0 --dataset "train"
python main.py --action "plot" --input "results/regr_weights/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_1" --num_classes "51" \
--training_method "regr_weights" --features Variant-2.0 --dataset "valid"


# Plots (regr_c012s)
python main.py --action "plot" --input "results/regr_c012s/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_4" --num_classes "51" \
--training_method "regr_c012s" --features Variant-2.0 --dataset "test" --use_filtered_data
python main.py --action "plot" --input "results/regr_c012s/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_4" --num_classes "51" \
--training_method "regr_c012s" --features Variant-2.0 --dataset "train" --use_filtered_data
python main.py --action "plot" --input "results/regr_c012s/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_4" --num_classes "51" \
--training_method "regr_c012s" --features Variant-2.0 --dataset "valid" --use_filtered_data

python main.py --action "plot" --input "results/regr_c012s/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_4" --num_classes "51" \
--training_method "regr_c012s" --features Variant-2.0 --dataset "test"
python main.py --action "plot" --input "results/regr_c012s/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_4" --num_classes "51" \
--training_method "regr_c012s" --features Variant-2.0 --dataset "train"
python main.py --action "plot" --input "results/regr_c012s/51_classes_variant_2.0/predictions" \
--output "plots/figures" --option "RESULTS_ANALYSIS_4" --num_classes "51" \
--training_method "regr_c012s" --features Variant-2.0 --dataset "valid"


# Plots (regr_argmaxs)
python main.py --action "plot" --input "results/regr_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_5" --num_classes "51" \
--training_method "regr_argmaxs" --features Variant-2.0 --dataset "test" --use_filtered_data
python main.py --action "plot" --input "results/regr_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_5" --num_classes "51" \
--training_method "regr_argmaxs" --features Variant-2.0 --dataset "train" --use_filtered_data
python main.py --action "plot" --input "results/regr_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_5" --num_classes "51" \
--training_method "regr_argmaxs" --features Variant-2.0 --dataset "valid" --use_filtered_data

python main.py --action "plot" --input "results/regr_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_5" --num_classes "51" \
--training_method "regr_argmaxs" --features Variant-2.0 --dataset "test"
python main.py --action "plot" --input "results/regr_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_5" --num_classes "51" \
--training_method "regr_argmaxs" --features Variant-2.0 --dataset "train"
python main.py --action "plot" --input "results/regr_argmaxs/51_classes_variant_2.0/predictions" \
--output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_5" --num_classes "51" \
--training_method "regr_argmaxs" --features Variant-2.0 --dataset "valid"

deactivate