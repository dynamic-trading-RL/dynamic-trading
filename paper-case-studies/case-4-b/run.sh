#!/bin/bash
#
#SBATCH --job-name=tr-4b
#SBATCH --cpus-per-task=50
python train_RL_agent.py && python out_of_sample.py && python backtesting.py