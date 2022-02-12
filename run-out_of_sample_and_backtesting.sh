#!/bin/bash
#
#SBATCH --job-name=trading
#SBATCH --cpus-per-task=50
python out_of_sample.py && python backtesting.py