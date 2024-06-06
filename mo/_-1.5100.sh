#!/bin/bash
#SBATCH --job-name              mo-1.5100
#SBATCH --time                  72:00:00   
#SBATCH --gpus                  1
#SBATCH --output                /home/mc35291/mo/log/_-1.5100.txt
#SBATCH --error                 /home/mc35291/mo/log/error.txt
#SBATCH --partition             gpu_batch
source activate torch2.0
python /home/mc35291/mo/main.py --snr1 -1.5 --snr2 100 --batchSize 2048 --totalbatch 20  --train 1 --core 4 --truncated 9 --seq_reloc 0
