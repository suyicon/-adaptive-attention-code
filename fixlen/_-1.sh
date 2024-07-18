#!/bin/bash
#SBATCH --job-name              fixlen-1100
#SBATCH --time                  96:00:00   
#SBATCH --gpus                  1
#SBATCH --output                /home/mc35291/repo/attn_code/fixlen/log/_-1.txt
#SBATCH --error                 /home/mc35291/repo/attn_code/fixlen/log/error.txt
#SBATCH --partition             gpu_batch
source activate torch2.0
python /home/mc35291/repo/attn_code/fixlen/main.py --snr1 -1 --snr2 100 --batchSize 2048 --totalbatch 40  --train 1 --core 4 --truncated 9 --reloc 0
