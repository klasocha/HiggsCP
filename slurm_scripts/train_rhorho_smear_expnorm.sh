#!/bin/bash -l
## Job name
#SBATCH -J TrainRhoRhoExpnormSmear
## Number of nodes
#SBATCH -N 1
## Number of tasks per node
#SBATCH --ntasks-per-node=1
## Memory per CPU (default: 5GB)
#SBATCH --mem-per-cpu=40GB
## Max job time (HH:MM:SS format)
#SBATCH --time=32:00:00
## Pratition specification
#SBATCH -p plgrid
#SBATCH --array=0-8

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
BETAS=(0.2 0.4 0.6 0.2 0.4 0.6 0.2 0.4 0.6)
SIGMAS=(0 0 0 0.2 0.2 0.2 0.4 0.4 0.4)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
SIGMA=${SIGMAS[$SLURM_ARRAY_TASK_ID]}

echo "TrainSmear rhorho Job. Beta: " $BETA
echo "Sigma: " $SIGMA

$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_rhorho -i $RHORHO_DATA -e 50 -f Variant-3.2 -d 0.2 --beta $BETA --smear_scale $SIGMA -l 6 -s 300
