#!/bin/bash

#SBATCH --job-name=variant_all_regr_c012s
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
python main.py --action "train" --input "data" --num_classes "51" --epochs $EPOCHS \
--training_method "regr_c012s" --model_location "51_classes_variant_all" \
--features Variant-All

# Prediction (test)
python main.py --action "predict_test" --input "data" --num_classes "51" \
--training_method "regr_c012s" --model_location "51_classes_variant_all" \
--features Variant-All --use_filtered_data
python main.py --action "predict_test" --input "data" --num_classes "51" \
--training_method "regr_c012s" --model_location "51_classes_variant_all" \
--features Variant-All

# Prediction (training and validation)
python main.py --action "predict_train_and_valid" --input "data" --num_classes "51" \
--training_method "regr_c012s" --model_location "51_classes_variant_all" \
--features Variant-All --use_filtered_data
python main.py --action "predict_train_and_valid" --input "data" --num_classes "51" \
--training_method "regr_c012s" --model_location "51_classes_variant_all" \
--features Variant-All

deactivate