#!/bin/bash
#
#SBATCH --job-name=tr-new
#SBATCH --cpus-per-task=10
python s_train_agent.py && python s_in_sample_tests.py