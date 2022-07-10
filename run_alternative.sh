#!/bin/bash
#
#SBATCH --job-name=tr-alt
#SBATCH --cpus-per-task=40
python s_train_agent.py && python s_backtesting.py