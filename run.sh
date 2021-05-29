#!/bin/bash
#
#SBATCH --job-name=trading
#SBATCH --cpus-per-task=40
python get_time_series.py && python build_strategies.py