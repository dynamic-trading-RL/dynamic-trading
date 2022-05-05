#!/bin/bash
#
#SBATCH --job-name=tr-oos-bkt-2
#SBATCH --cpus-per-task=20
python out_of_sample.py && python backtesting.py