#!/bin/bash
#
#SBATCH --job-name=trading
#SBATCH --cpus-per-task=50
python train_RL_agent.py && python backtesting.py && python out_of_sample.py