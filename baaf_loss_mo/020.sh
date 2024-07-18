#!/bin/bash
#SBATCH --job-name              baaf_loss_mo_020
#SBATCH --time                  96:00:00   
#SBATCH --gpus                  1
#SBATCH --output                /home/mc35291/repo/attn_code/baaf_loss_mo/log/020.txt
#SBATCH --error                 /home/mc35291/repo/attn_code/baaf_loss_mo/log/error.txt
#SBATCH --partition             gpu_batch
source activate torch2.0
python /home/mc35291/repo/attn_code/baaf_loss_mo/main.py --snr1 0 --snr2 20 --batchSize 2048 --totalbatch 40  --train 1 --core 4 --truncated 10 --reloc 0
