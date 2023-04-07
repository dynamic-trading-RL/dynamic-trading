#!/bin/bash
#
#SBATCH --job-name=tr-2
#SBATCH --cpus-per-task=20
python fit_dynamics.py && python train_RL_agent.py && python out_of_sample.py && python backtesting.py