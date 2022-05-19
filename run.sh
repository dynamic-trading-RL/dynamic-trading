#!/bin/bash
#
#SBATCH --job-name=tr-new
#SBATCH --cpus-per-task=20
python scripts/s_train_agent.py && python scripts/s_in_sample_tests.py