#!/bin/bash
#SBATCH --job-name=gliner-ner
#SBATCH --account=def-cepp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/output_%A_%a.log
#SBATCH --mail-user=rany@ualberta.ca
#SBATCH --mail-type=BEGIN,END

echo "Starting job ${SLURM_ARRAY_TASK_ID}"

nvidia-smi

module load python/3.11.5
module load cuda/12.6
source cc/bin/activate

# Create output folder
mkdir -p outputs

echo "Running split ${SLURM_ARRAY_TASK_ID}"
python your_script.py ${SLURM_ARRAY_TASK_ID}

echo "Finished split ${SLURM_ARRAY_TASK_ID}"