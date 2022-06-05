#!/bin/bash
#
#SBATCH --job-name=tr-new
#SBATCH --cpus-per-task=10
python s_calibrate_market_dynamics.py && python s_train_agent.py && python s_backtesting.py