#!/bin/bash
ulimit -s unlimited 
#SBATCH --partition=H100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=20
#SBATCH --gpus=3
#SBATCH --ntasks=3
#SBATCH --array=0-2
 
# Define the array of image counts
CFG_PATHS=(
    'projects/deformable_detr/configs/deformable_detr_r50_50ep_pheno.py' 
    'projects/deformable_detr/configs/deformable_detr_r50_50ep_syn_v6.py'
    'projects/deformable_detr/configs/deformable_detr_r50_50ep_syn_v6_augmented.py'
    )
 
# Get the current image count based on the array task ID
CURRENT_CFG_PATHS=${CFG_PATHS[$SLURM_ARRAY_TASK_ID]}

SUBSTRING=$(echo $CURRENT_CFG_PATHS| cut -d'_' -f 6)
# Set the job name dynamically
SLURM_JOB_NAME="Def_DETR_${SUBSTRING}"
#SBATCH --job-name=$SLURM_JOB_NAME
 
# Run the experiment
srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=20 -p H100 --mem=40000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/detrex_orig:/home/iqbal/detrex \
  --container-image=/netscratch/naeem/detrex-torch1.13-detectron2-sourced.sqsh \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=$SLURM_JOB_NAME \
  --container-workdir=/home/iqbal/detrex \
  --time=00-01:00 \
  bash train_detrex.sh $CURRENT_CFG_PATHS