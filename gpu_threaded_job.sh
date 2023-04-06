#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-gregorys
#SBATCH --gres=gpu:1        # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=63500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH --time=06:00:00
#SBATCH --job-name=fatema_test1
#SBATCH --output=some_name-%j.out
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# activate your virtual environment
source /home/fatema/ENV/bin/activate

# load necessary modules
module load python/3.10
nvidia-smi

# run your python script with parameters
python -u run_CCST_edited.py --data_name=exp1_V10M25_60_C1_140694_Spatial10X --data_path=generated_data_pca/ --num_epoch=15000 --hidden=256 --model_name=exp1_V10M25_60_C1_140694_Spatial10X_test1 --GNN_type='GATConv' 
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
