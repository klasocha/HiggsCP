#!/bin/bash

#SBATCH --job-name=variant_1.0_results_analysis
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

# Unweighted events test
python main.py --action "test_model_on_unwt_events" --input "data_new_format" \
--output "plots/figures/test_model_on_unwt_events/51_classes_variant_1.0/soft_weights" --num_classes "51" \
--hypothesis "0-4-46" --training_method "soft_weights" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0" --data_format "v2"

python main.py --action "test_model_on_unwt_events" --input "data_new_format" \
--output "plots/figures/test_model_on_unwt_events/51_classes_variant_1.0/regr_weights" --num_classes "51" \
--hypothesis "0-4-46" --training_method "regr_weights" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0" --data_format "v2"

python main.py --action "test_model_on_unwt_events" --input "data_new_format" \
--output "plots/figures/test_model_on_unwt_events/51_classes_variant_1.0/soft_c012s" --num_classes "51" \
--hypothesis "0-4-46" --training_method "soft_c012s" --model_location "variant_1.0/51_classes" \
--features "Variant-1.0" --data_format "v2"

python main.py --action "test_model_on_unwt_events" --input "data_new_format" \
--output "plots/figures/test_model_on_unwt_events/51_classes_variant_1.0/regr_c012s" --num_classes "51" \
--hypothesis "0-4-46" --training_method "regr_c012s" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0" --data_format "v2"

python main.py --action "test_model_on_unwt_events" --input "data_new_format" \
--output "plots/figures/test_model_on_unwt_events/51_classes_variant_1.0/soft_argmaxs" --num_classes "51" \
--hypothesis "0-4-46" --training_method "soft_argmaxs" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0"

python main.py --action "test_model_on_unwt_events" --input "data_new_format" \
--output "plots/figures/test_model_on_unwt_events/51_classes_variant_1.0/regr_argmaxs" --num_classes "51" \
--hypothesis "0-4-46" --training_method "regr_argmaxs" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0" --keras "v2"


# All events test
python main.py --action "test_model_on_all_events" --input "data_new_format" \
--output "plots/figures/test_model_on_all_events/51_classes_variant_1.0/soft_weights" --num_classes "51" \
--training_method "soft_weights" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0"

python main.py --action "test_model_on_all_events" --input "data_new_format" \
--output "plots/figures/test_model_on_all_events/51_classes_variant_1.0/regr_weights" --num_classes "51" \
--training_method "regr_weights" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0"

python main.py --action "test_model_on_all_events" --input "data_new_format" \
--output "plots/figures/test_model_on_all_events/51_classes_variant_1.0/soft_c012s" --num_classes "51" \
--training_method "soft_c012s" --model_location "variant_1.0/51_classes" \
--features "Variant-1.0"

python main.py --action "test_model_on_all_events" --input "data_new_format" \
--output "plots/figures/test_model_on_all_events/51_classes_variant_1.0/regr_c012s" --num_classes "51" \
--training_method "regr_c012s" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0"

python main.py --action "test_model_on_all_events" --input "data_new_format" \
--output "plots/figures/test_model_on_all_events/51_classes_variant_1.0/soft_argmaxs" --num_classes "51" \
--training_method "soft_argmaxs" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0"

python main.py --action "test_model_on_all_events" --input "data_new_format" \
--output "plots/figures/test_model_on_all_events/51_classes_variant_1.0/regr_argmaxs" --num_classes "51" \
--training_method "regr_argmaxs" --model_location "51_classes_variant_1.0" \
--features "Variant-1.0" --keras "v2"

deactivate