#!/bin/bash
#Here is a comment
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=HimiIsNERing
#SBATCH --account=rcc-staff
#SBATCH --output=HimiIsNERing-%j.out
#SBATCH --error=HimiIsNERing-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=BEGIN
export PYTHONUNBUFFERED=TRUE
set ORIG_LD_LIBRARY_PATH $::env"ORIG_LD_LIBRARY_PATH"
exec logger -t modulecmd "[get-env USER]: $command [join $args]"
set ::env"ORIG_LD_LIBRARY_PATH" $ORIG_LD_LIBRARY_PATH
module unload pytorch/1.10
module unload python
module load python/anaconda-2022.05
module load cuda/10.2
source activate spacy_venv
python3 -u -m spacy train data/config.cfg --paths.train ./train.spacy --paths.dev ./valid.spacy --output ./models/output --gpu-id 0
