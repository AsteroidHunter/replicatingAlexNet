#!/bin/bash
#SBATCH --job-name=full_training_run
#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --time=80:00:00
#SBATCH --account=<removed>
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=6gb
#SBATCH --partition=gpu_standard

module load python/3.11/3.11.4 cuda11 cuda11-dnn cuda11-sdk ohpc
python3 alexnet_torch_hpc_run.py