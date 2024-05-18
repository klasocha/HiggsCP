#!/bin/bash

export EPOCHS=120

JOBID0=$(sbatch --parsable -p plgrid other/slurm_scripts/variant_1.0/0_prepare_data.sh)
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.0/1_soft_weights.sh
JOBID2=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.0/2_soft_c012s_c0.sh)
JOBID3=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.0/3_soft_c012s_c1.sh)
sbatch --dependency afterok:$JOBID0:$JOBID2:$JOBID3 -p plgrid other/slurm_scripts/variant_1.0/4_soft_c012s_c2.sh
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.0/5_soft_argmaxs.sh
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.0/6_regr_weights.sh
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.0/7_regr_c012s.sh
sbatch --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/variant_1.0/8_regr_argmaxs.sh

JOBID4=$(sbatch --parsable -p plgrid other/slurm_scripts/variant_4.1/0_prepare_data.sh)
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_4.1/1_soft_weights.sh
JOBID5=$(sbatch --parsable --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_4.1/2_soft_c012s_c0.sh)
JOBID6=$(sbatch --parsable --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_4.1/3_soft_c012s_c1.sh)
sbatch --dependency afterok:$JOBID4:$JOBID5:$JOBID6 -p plgrid other/slurm_scripts/variant_4.1/4_soft_c012s_c2.sh
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_4.1/5_soft_argmaxs.sh
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_4.1/6_regr_weights.sh
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_4.1/7_regr_c012s.sh
sbatch --dependency afterok:$JOBID4 -p plgrid other/slurm_scripts/variant_4.1/8_regr_argmaxs.sh