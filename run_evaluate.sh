#!/bin/bash

#SBATCH --cpus-per-task=31
#SBATCH --mem=9G        # memory per node
#SBATCH --account=def-vganesh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luzhengyang1231@outlook.com
#SBATCH --time=0-6:00      # time (DD-HH:MM)
#SBATCH --output=run-%N-%j.out  # %N for node name, %j for jobID

module load python/3.7

source ~/env/alphasmt/bin/activate

export PYTHONUNBUFFERED=TRUE

python evaluate.py eval/eval_apr4_cinteger.json