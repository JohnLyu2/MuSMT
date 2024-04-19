#!/bin/bash

#SBATCH --cpus-per-task=26
#SBATCH --mem=12G        # memory per node
#SBATCH --account=def-vganesh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luzhengyang1231@outlook.com
#SBATCH --time=0-24:00      # time (DD-HH:MM)
#SBATCH --output=run-%N-%j.out  # %N for node name, %j for jobID

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

OUT_FOLD="exp_results/$(timestamp)"
mkdir -p "$OUT_FOLD"
echo "Writing output to folder $OUT_FOLD"

module load python/3.7

source ~/alphasmt/bin/activate

export PYTHONUNBUFFERED=TRUE

python main.py experiments/cinteger20_apr23.json \
    > "$OUT_FOLD/out.txt" 2>&1