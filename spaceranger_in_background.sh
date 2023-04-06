#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for running job in background.
# ---------------------------------------------------------------------
#SBATCH -c 12
#SBATCH --mem=30GB
#SBATCH --time=5:00:00
#SBATCH --job-name=spaceranger
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
# cd to your desired directory
cd /cluster/projects/schwartzgroup/fatema/pancreatic_cancer_visium
echo "Current working directory: `pwd`"
# load necessary modules
module load spaceranger/2.0.0
# run your command with parameters
spaceranger count --id=V10M25-61_D1_PDA_64630_Pa_P_Spatial10x_new --sample PDA_64630_Pa_P_Spatial10x --transcriptome /cluster/projects/schwartzgroup/fatema/refdata-gex-GRCh38-2020-A --fastqs /cluster/projects/schwartzgroup/data/notta_pancreatic_cancer_visium/210827_A00827_0396_BHJLJTDRXY_Notta_Karen/ --image /cluster/home/t116508uhn/data/notta_pancreatic_cancer_visium/V10M25_61_D1_Exp2.tif --loupe-alignment /cluster/home/t116508uhn/64630/V10M25-061-D1.json --slide V10M25-061 --area D1 --localmem 20 


# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"


