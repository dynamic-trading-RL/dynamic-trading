#!/bin/bash
#
#SBATCH --job-name=trading
#SBATCH --cpus-per-task=50
python train_RL_agent.py && out_of_sample.py