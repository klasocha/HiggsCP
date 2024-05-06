#!/bin/bash

#SBATCH --job-name=variant_1.1_soft_c012s_c2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=9216Mb
#SBATCH --time=02:50:00
#SBATCH --output="results/slurm_scripts_logs/output-%A_%a.out"
#SBATCH --error="results/slurm_scripts_logs/error-%A_%a.err"

cd $SLURM_SUBMIT_DIR
module load python/3.11.3-gcccore-12.3.0
source .venv/bin/activate

python main.py --action "download_and_preprocess" --input "data_c2" --features Variant-1.1 --num_classes "51" --hits_c012s "hits_c2s"
python main.py --action "train" --input "data_c2" --num_classes "51" --epochs "120" --training_method "soft_c012s" --model_location "variant_1.1/51_classes_c2" --hits_c012s "hits_c2s"
python main.py --action "predict_test" --input "data_c2" --num_classes "51" --training_method "soft_c012s" --model_location "variant_1.1/51_classes_c2" --training_method "soft_c012s" --dataset "test" --use_filtered_data
python main.py --action "predict_train_and_valid" --input "data_c2" --num_classes "51" --training_method "soft_c012s" --model_location "variant_1.1/51_classes_c2" --training_method "soft_c012s" --dataset "valid" --use_filtered_data

python main.py --action "plot" --input "results/soft_c012s/variant_1.1/51_classes_c" --output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_2" --num_classes "51" --training_method "soft_c012s" --features Variant-1.1 --dataset "test" --use_filtered_data
python main.py --action "plot" --input "results/soft_c012s/variant_1.1/51_classes_c" --output "plots/figures" --format "pdf" --option "RESULTS_ANALYSIS_2" --num_classes "51" --training_method "soft_c012s" --features Variant-1.1 --dataset "test" --use_filtered_data
python main.py --action "plot" --input "results/soft_c012s/variant_1.1/51_classes_c" --output "plots/figures" --format "eps" --option "RESULTS_ANALYSIS_2" --num_classes "51" --training_method "soft_c012s" --features Variant-1.1 --dataset "test" --use_filtered_data

python main.py --action "plot" --input "results/soft_c012s/variant_1.1/51_classes_c" --output "plots/figures" --format "png" --option "RESULTS_ANALYSIS_2" --num_classes "51" --training_method "soft_c012s" --features Variant-1.1 --dataset "valid" --use_filtered_data
python main.py --action "plot" --input "results/soft_c012s/variant_1.1/51_classes_c" --output "plots/figures" --format "pdf" --option "RESULTS_ANALYSIS_2" --num_classes "51" --training_method "soft_c012s" --features Variant-1.1 --dataset "valid" --use_filtered_data
python main.py --action "plot" --input "results/soft_c012s/variant_1.1/51_classes_c" --output "plots/figures" --format "eps" --option "RESULTS_ANALYSIS_2" --num_classes "51" --training_method "soft_c012s" --features Variant-1.1 --dataset "valid" --use_filtered_data

deactivate