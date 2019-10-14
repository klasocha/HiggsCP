#!/bin/bash -l
## Job name
#SBATCH -J TrainBetaA1RHONoDrop
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
#SBATCH --array=0-2

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
BETAS=(0.2 0.4 0.6)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "TrainBeta a1rho NODROP Job. Beta: " $BETA

$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_a1rho -i $A1RHO_DATA -e 50 -f Variant-3.1 -d 0 --beta $BETA -l 6 -s 300
