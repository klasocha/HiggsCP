#!/bin/bash

#SBATCH -job-name=Compute_C012_for_all_hypotheses
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3850M
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid
#SBATCH --array=0-10
#SBATCH --output="results/output-%A_%a.out"
#SBATCH --error="results/error-%A_%a.err"

cd $SLURM_SUBMIT_DIR

## Command
HYPOTHESES=(00 02 04 06 08 10 12 14 16 18 20)
ALPHA_CP_CLASS=${HYPOTHESES[$SLURM_ARRAY_TASK_ID]}
echo "Computing the C0/C1/C2 coefficients for the hypothesis specified by alphaCP=" $ALPHA_CP_CLASS
python unweight_events.py --input "data" --option "PREPARE-C012S" --hypothesis $ALPHA_CP_CLASS