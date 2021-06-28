#!/bin/bash
#
#SBATCH --job-name=trading
#SBATCH --cpus-per-task=80
python back_testing.py && python out_of_sample.py