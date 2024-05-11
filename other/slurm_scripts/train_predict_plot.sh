#!/bin/bash

EPOCHS=120

JOBID0=$(sbatch --parsable -p plgrid other/slurm_scripts/variant_1.1/0_prepare_data.sh --export=EPOCHS)
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.1/1_train_soft_weights.sh --export=EPOCHS
JOBID2=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.1/2_soft_c012s_c0.sh --export=EPOCHS)
JOBID3=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.1/3_soft_c012s_c1.sh --export=EPOCHS)
sbatch --dependency afterok:$JOBID0:$JOBID2:$JOBID3 -p plgrid other/slurm_scripts/variant_1.1/4_soft_c012s_c2.sh --export=EPOCHS
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.1/5_soft_argmaxs.sh --export=EPOCHS
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.1/6_regr_weights.sh --export=EPOCHS
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.1/7_regr_c012s.sh --export=EPOCHS
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.1/8_regr_argmaxs.sh --export=EPOCHS

JOBID4=$(sbatch --parsable -p plgrid other/slurm_scripts/variant_all/0_prepare_data.sh --export=EPOCHS)
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_all/1_train_soft_weights.sh --export=EPOCHS
JOBID5=$(sbatch --parsable --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_all/2_soft_c012s_c0.sh --export=EPOCHS)
JOBID6=$(sbatch --parsable --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_all/3_soft_c012s_c1.sh --export=EPOCHS)
sbatch --dependency afterok:$JOBID4:$JOBID5:$JOBID6 -p plgrid other/slurm_scripts/variant_all/4_soft_c012s_c2.sh --export=EPOCHS
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_all/5_soft_argmaxs.sh --export=EPOCHS
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_all/6_regr_weights.sh --export=EPOCHS
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_all/7_regr_c012s.sh --export=EPOCHS
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_all/8_regr_argmaxs.sh --export=EPOCHS