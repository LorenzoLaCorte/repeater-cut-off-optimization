#!/bin/bash
#SBATCH -p compute
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH -c 8

python3 distillation_ml_gp.py
