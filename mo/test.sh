#!/bin/bash
#SBATCH --job-name              test_mo10
#SBATCH --time                  72:00:00
#SBATCH --gpus                  1
#SBATCH --output                /home/mc35291/mo/log/test.txt
#SBATCH --error                 /home/mc35291/mo/log/error.txt
#SBATCH --partition             gpu_batch
source activate torch2.0
python /home/mc35291/mo/main.py --snr1 0 --snr2 100 --batchSize 100000 --totalbatch 10  --train 0 --core 1 --truncated 10 --reloc 0