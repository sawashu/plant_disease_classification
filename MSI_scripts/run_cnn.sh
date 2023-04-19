#!/bin/bash -l        
#SBATCH --time=6:00:00
#SBATCH --ntasks=8
#SBATCH --mem=32g
#SBATCH --tmp=32g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=uribe055@umn.edu 

cnn_directory = /global/work/raine124/cnn_slurm_job
mkdir -p cnn_directory
cp /Users/bean/Documents/plant_disease_classification cnn_directory
cd cnn_directory/plant_disease_classification/Run_Models
module load python3
python3 Run_CNN_Models.py
cp cnn_directory /Users/bean/Documents/plant_disease_classification
cd /Users/bean/Documents/plant_disease_classification
rm -rf cnn_directory