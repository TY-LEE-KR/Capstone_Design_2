#!/bin/bash

#SBATCH --job-name=dualprompt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch_ce_ugrad
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time=1-00:00:00
#SBATCH -o %x_%j.out
#SBTACH -e %x_%j.err

python  main.py \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5

exit 0