#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=f.daniotti@student.tue.nl

# go into your project folder
cd $TMPDIR/project/5AUA0-Project4

# activate venv
source venv/bin/activate

# make sure on correct branch
git checkout master

# run your code
python3 train.py -c hmc_gpu_3ch_e30