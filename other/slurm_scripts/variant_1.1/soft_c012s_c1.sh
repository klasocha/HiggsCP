#!/bin/bash

#SBATCH --job-name=Train_predict_plot_for_variant_1.1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3850M
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid
#SBATCH --output="results/output-%A_%a.out"
#SBATCH --error="results/error-%A_%a.err"

cd $SLURM_SUBMIT_DIR
python main.py --action "download_and_preprocess" --input "data" --features Variant-1.1 --num_classes "51" --hits_c012s "hits_c1s"
python main.py --action "train" --input "data" --num_classes "51" --epochs "120" --training_method "soft_c012s" --model_location "51_classes_c1" --hits_c012s "hits_c1s"
python main.py --action "predict_test" --input "data" --num_classes "51" --training_method "soft_c012s" --model_location "51_classes_c1" --training_method "soft_c012s" --dataset "test" --use_filtered_data