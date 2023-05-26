#!/bin/bash
#Here is a comment
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=HimiIsTesting
#SBATCH --account=rcc-staff
#SBATCH --output=./outputs/ML-%j.out
#SBATCH --error=./outputs/ML-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=BEGIN

module load python/anaconda-2022.05

source activate markuplmft

module unload python/anaconda-2022.05

python3 Markup_LM_native_pytorch.py

