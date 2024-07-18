#!/bin/bash
#SBATCH --job-name              gradnorm_mo10
#SBATCH --time                  48:00:00   
#SBATCH --gpus                  1
#SBATCH --output                /home/mc35291/repo/attn_code/gradnorm_mo/log/0100.txt
#SBATCH --error                 /home/mc35291/repo/attn_code/gradnorm_mo/log/error.txt
#SBATCH --partition             a100_batch
source activate torch2.0
python /home/mc35291/repo/attn_code/gradnorm_mo/main.py --snr1 0 --snr2 100 --batchSize 4096 --totalbatch 10  --train 1 --core 1 --truncated 10 --reloc 0
