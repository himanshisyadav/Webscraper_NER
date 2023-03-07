#!/bin/bash
#Here is a comment
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=HimiIsTesting
#SBATCH --account=rcc-staff
#SBATCH --output=ML-%j.out
#SBATCH --error=ML-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=BEGIN

module load python/anaconda-2022.05

source activate markuplmft

python3 Markup_LM_native_pytorch.py

