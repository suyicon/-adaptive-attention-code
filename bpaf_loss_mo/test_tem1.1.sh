#!/bin/bash
#SBATCH --job-name              test_tem1.1_mo10
#SBATCH --time                  72:00:00
#SBATCH --gpus                  1
#SBATCH --output                /home/mc35291/repo/attn_code/mo/log/test_t1.1.txt
#SBATCH --error                 /home/mc35291/repo/attn_code/mo/log/error.txt
#SBATCH --partition             gpu_batch
source activate torch2.0
python /home/mc35291/repo/attn_code/mo/main.py --temp 1.1 --snr1 0 --snr2 100 --batchSize 100000 --totalbatch 10  --train 0 --core 1 --truncated 10 --reloc 0