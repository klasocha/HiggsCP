#!/bin/bash

export EPOCHS=25
export EXP=RhoRho

JOBID0=$(sbatch --parsable -p plgrid other/slurm_scripts/rhorho/variant_all/0_prepare_data.sh)
JOBID1=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_all/1_soft_weights.sh)
JOBID2=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_all/2_soft_c012s_c0.sh)
JOBID3=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_all/3_soft_c012s_c1.sh)
JOBID4=$(sbatch --parsable --dependency afterok:$JOBID0:$JOBID2:$JOBID3 -p plgrid other/slurm_scripts/rhorho/variant_all/4_soft_c012s_c2.sh)
JOBID5=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_all/5_soft_argmaxs.sh)
JOBID6=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_all/6_regr_weights.sh)
JOBID7=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_all/7_regr_c012s.sh)
JOBID8=$(sbatch --parsable --dependency afterok:$JOBID0 -p plgrid other/slurm_scripts/rhorho/variant_all/8_regr_argmaxs.sh)
JOBID9=$(sbatch --parsable --dependency \
afterok:$JOBID0:$JOBID1:$JOBID2:$JOBID3:$JOBID4:$JOBID5:$JOBID6:$JOBID7:$JOBID8 \
-p plgrid other/slurm_scripts/rhorho/variant_all/9_results_analysis.sh)
JOBID10=$(sbatch --parsable --dependency \
afterok:$JOBID0:$JOBID1:$JOBID2:$JOBID3:$JOBID4:$JOBID5:$JOBID6:$JOBID7:$JOBID8 \
-p plgrid other/slurm_scripts/rhorho/variant_all/10_test_evt.sh)