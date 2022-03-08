#!/bin/bash
#
#SBATCH --job-name=tr-oos-bkt-4b
#SBATCH --cpus-per-task=20
python out_of_sample.py && python backtesting.py