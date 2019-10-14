#!/bin/bash -l
## Job name
#SBATCH -J TrainDropoutFig7
## Number of nodes
#SBATCH -N 1
## Number of tasks per node
#SBATCH --ntasks-per-node=1
## Memory per CPU (default: 5GB)
#SBATCH --mem-per-cpu=40GB
## Max job time (HH:MM:SS format)
#SBATCH --time=72:00:00
## Pratition specification
#SBATCH -p plgrid
#SBATCH --array=0-1

## Setup
if [ -f setup ]; then source setup; fi
export PYTHONPATH=$ANACONDA_PYTHON_PATH
module add plgrid/apps/cuda/7.5

## Chaninging dir to workdir
cd $WORKDIR

## Command
FEATURE_SETS=(0 0.2)
FEATURE_SET=${FEATURE_SETS[$SLURM_ARRAY_TASK_ID]}
echo "TrainDROPOUTA1Rho Job. Fig 7. DROPOUT: " $FEATURE_SET
$ANACONDA_PYTHON_PATH/python2.7 $WORKDIR/main.py -t nn_a1rho -i $A1RHO_DATA -e 250 -f Variant-1.1 -d $FEATURE_SET -l 6 -s 300
