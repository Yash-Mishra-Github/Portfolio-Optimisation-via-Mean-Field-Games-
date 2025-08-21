#!/bin/bash
#SBATCH --job-name=quick_tuned
#SBATCH --output=logs/quick_tuned_%j.out
#SBATCH --error=logs/quick_tuned_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --qos=long

module load Anaconda3
source activate mfg_env

echo "Running Stage I tuned (train_quick.py, Transformer, longer run)"
srun python -u src/train_quick.py \
  --model transformer \
  --data data/processed_market_data.csv \
  --out outputs/quick_tuned_${SLURM_JOB_ID} \
  --iters 3000 \
  --N 512 \
  --K 252 \
  --lr 0.0003
