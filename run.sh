#!/bin/bash
#
#SBATCH --job-name=trading
#SBATCH --cpus-per-task=80
python get_time_series.py && python train_RL_agent.py && python back_testing.py && python out_of_sample.py