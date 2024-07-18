#!/bin/bash
#SBATCH --job-name              mo10tem11
#SBATCH --time                  72:00:00   
#SBATCH --gpus                  1
#SBATCH --output                /home/mc35291/repo/attn_code/mo/log/0100.txt
#SBATCH --error                 /home/mc35291/repo/attn_code/mo/log/error.txt
#SBATCH --partition             gpu_batch
source activate torch2.0
python /home/mc35291/repo/attn_code/mo/main.py --temp 1.1 --snr1 0 --snr2 100 --batchSize 2048 --totalbatch 40  --train 1 --core 4 --truncated 10 --reloc 0
