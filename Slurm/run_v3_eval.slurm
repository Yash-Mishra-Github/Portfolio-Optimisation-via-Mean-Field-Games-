#!/bin/bash
#
# Fabian overnight evaluation job for V3 (UNIX line endings)
#
#SBATCH -J v3_eval_sigdfp_bench
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /users/mishray/V3/logs/eval_%j.out
#SBATCH -e /users/mishray/V3/logs/eval_%j.err

set -euo pipefail
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sigdfp_env
set -u
python --version
which python

echo "== Node & Env =="
hostname
date
python --version
which python

# Ensure dirs
mkdir -p /users/mishray/V3/logs
mkdir -p /users/mishray/V3/results
mkdir -p /users/mishray/V3/data
mkdir -p /users/mishray/V3/outputs

# Allow overrides through env without editing the file
OUTPUTS_ROOTS="${OUTPUTS_ROOTS:-/users/mishray/V3/outputs /users/mishray/outputs /users/mishray/outputs_V2}"
DATA_FILE="${DATA_FILE:-/users/mishray/V3/data/processed_market_data.csv}"
RES_ROOT="${RES_ROOT:-/users/mishray/V3/results}"
WINDOW="${WINDOW:-126}"
TAG="${TAG:-overnight}"
INCLUDE="${INCLUDE:-}"
EXCLUDE="${EXCLUDE:-}"
MAX_RUNS="${MAX_RUNS:-0}"

python /users/mishray/V3/evaluate_v3.py \
  --outputs-root ${OUTPUTS_ROOTS} \
  --data-file "${DATA_FILE}" \
  --window ${WINDOW} \
  --res-root "${RES_ROOT}" \
  --tag "${TAG}" \
  ${INCLUDE:+--include "${INCLUDE}"} \
  ${EXCLUDE:+--exclude "${EXCLUDE}"} \
  --max-runs ${MAX_RUNS}

echo "== Job complete =="
date
