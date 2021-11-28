#!/bin/bash
#
#SBATCH --job-name=trading
#SBATCH --cpus-per-task=50
python train_RL_agent.py