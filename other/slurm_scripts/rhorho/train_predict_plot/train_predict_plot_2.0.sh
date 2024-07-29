#!/bin/bash

export EPOCHS=25
export EXP=RhoRho

JOBID0=$(sbatch --parsable -p plgrid other/slurm_scripts/rhorho/variant_2.0/0_prepare_data.sh)
JOBID1=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_2.0/1_soft_weights.sh)
JOBID2=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_2.0/2_soft_c012s_c0.sh)
JOBID3=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_2.0/3_soft_c012s_c1.sh)
JOBID4=$(sbatch --parsable --dependency afterok:$JOBID0:$JOBID2:$JOBID3 -p plgrid other/slurm_scripts/rhorho/variant_2.0/4_soft_c012s_c2.sh)
JOBID5=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_2.0/5_soft_argmaxs.sh)
JOBID6=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_2.0/6_regr_weights.sh)
JOBID7=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_2.0/7_regr_c012s.sh)
JOBID8=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_2.0/8_regr_argmaxs.sh)
JOBID9=$(sbatch --parsable --dependency \
afterok:$JOBID0:$JOBID1:$JOBID2:$JOBID3:$JOBID4:$JOBID5:$JOBID6:$JOBID7:$JOBID8 \
-p plgrid other/slurm_scripts/rhorho/variant_2.0/9_results_analysis.sh)
JOBID10=$(sbatch --parsable --dependency \
afterok:$JOBID0:$JOBID1:$JOBID2:$JOBID3:$JOBID4:$JOBID5:$JOBID6:$JOBID7:$JOBID8 \
-p plgrid other/slurm_scripts/rhorho/variant_2.0/10_test_evt.sh)