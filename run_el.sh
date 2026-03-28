#!/bin/bash
#SBATCH --job-name=genre-el
#SBATCH --account=def-cepp
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/output_el_%A_%a.log
#SBATCH --mail-user=rany@ualberta.ca
#SBATCH --mail-type=BEGIN,END

echo "Starting EL job ${SLURM_ARRAY_TASK_ID}"
nvidia-smi

module load python/3.11.5
module load cuda/12.6
source cc/bin/activate

mkdir -p el_outputs
mkdir -p el_progress

echo "Running EL on split ${SLURM_ARRAY_TASK_ID}"
python perform_el.py ${SLURM_ARRAY_TASK_ID}
echo "Finished split ${SLURM_ARRAY_TASK_ID}"