#!/bin/bash

python main.py --action "download_and_prepare_original" --input "data_z" --exp "Z"
scp data/unwt_multiclass_51.npy data_z/
./other/slurm_scripts/z_background/train_predict_plot/train_predict_plot_1.0.sh
./other/slurm_scripts/z_background/train_predict_plot/train_predict_plot_1.1.sh
./other/slurm_scripts/z_background/train_predict_plot/train_predict_plot_2.0.sh
./other/slurm_scripts/z_background/train_predict_plot/train_predict_plot_2.1.sh
./other/slurm_scripts/z_background/train_predict_plot/train_predict_plot_4.1.sh
./other/slurm_scripts/z_background/train_predict_plot/train_predict_plot_all.sh