#!/bin/bash

sbatch -p plgrid other/slurm_scripts/z_background/all_events_variant_all.sh
sbatch -p plgrid other/slurm_scripts/z_background/all_events_variant_1.0.sh
sbatch -p plgrid other/slurm_scripts/z_background/all_events_variant_1.1.sh
sbatch -p plgrid other/slurm_scripts/z_background/all_events_variant_2.0.sh
sbatch -p plgrid other/slurm_scripts/z_background/all_events_variant_2.1.sh
sbatch -p plgrid other/slurm_scripts/z_background/all_events_variant_4.1.sh