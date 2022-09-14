#!/bin/bash
#
#SBATCH --job-name=tr
#SBATCH --cpus-per-task=20
python s_calibrate_specific_market_dynamics.py && python s_compute_shares_scale.py && python s_train_agent.py && python s_backtesting.py && python s_simulationtesting.py