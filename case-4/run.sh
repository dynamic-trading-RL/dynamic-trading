#!/bin/bash
#
#SBATCH --job-name=trad-case-4
#SBATCH --cpus-per-task=20
python train_RL_agent.py && python out_of_sample.py && python backtesting.py