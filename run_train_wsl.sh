#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export POKEMON_TRAIN_DEVICE="${POKEMON_TRAIN_DEVICE:-cuda}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export POKEMON_VEC_ENV="${POKEMON_VEC_ENV:-subproc}"
export POKEMON_SUBPROC_START_METHOD="${POKEMON_SUBPROC_START_METHOD:-forkserver}"

echo "Using POKEMON_TRAIN_DEVICE=${POKEMON_TRAIN_DEVICE}"
echo "Using HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo "Using POKEMON_VEC_ENV=${POKEMON_VEC_ENV}"
echo "Using POKEMON_SUBPROC_START_METHOD=${POKEMON_SUBPROC_START_METHOD}"

# Quick GPU sanity check
python3 -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print(f'GPU OK: {torch.cuda.get_device_name(0)}')"

exec python3 train.py
