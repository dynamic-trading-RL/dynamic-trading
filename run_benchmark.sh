#!/bin/bash
#
#SBATCH --job-name=tr-bmk
#SBATCH --cpus-per-task=20
python s_train_agent.py && python s_backtesting.py