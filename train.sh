#!/bin/bash

# Set job requirements
SBATCH --nodes=1
SBATCH --gpus-per-node=1
SBATCH --partition=gpu
SBATCH --time=5:00
SBATCH --mail-type=BEGIN,END
SBATCH --mail-user=f.daniotti@student.tue.nl

# go into your project folder
cd $TMPDIR/project/5AUA0-Project4

# activate venv
source venv/bin/activate

# make sure on correct branch
git checkout master

# run your code
python3 run.py -c $1
