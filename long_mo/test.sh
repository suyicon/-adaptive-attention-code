#!/bin/bash
#SBATCH --job-name              longmo_test
#SBATCH --time                  24:00:00   
#SBATCH --gpus                  1
#SBATCH --output                /home/mc35291/repo/attn_code/long_mo/log/test.txt
#SBATCH --error                 /home/mc35291/repo/attn_code/long_mo/log/error.txt
#SBATCH --partition             gpu_batch
source activate torch2.0
python /home/mc35291/repo/attn_code/long_mo/main.py --snr1 0 --snr2 100 --batchSize 16384 --totalbatch 40  --train 0 --core 1 --reloc 0 --truncated 15 --desired_T 10
