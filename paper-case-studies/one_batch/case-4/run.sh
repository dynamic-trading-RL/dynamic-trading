#!/bin/bash
#
#SBATCH --job-name=tr-4
#SBATCH --cpus-per-task=20
python train_RL_agent_stock.py && python out_of_sample_stock.py