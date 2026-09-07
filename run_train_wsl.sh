#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export POKEMON_VEC_ENV="${POKEMON_VEC_ENV:-subproc}"
export POKEMON_SUBPROC_START_METHOD="${POKEMON_SUBPROC_START_METHOD:-forkserver}"

# Device detection: honor an explicit POKEMON_TRAIN_DEVICE, otherwise probe
# for a working GPU and fall back to CPU. Machines without a GPU (e.g. the
# OVH host) just get device=cpu -- train.py's own _default_device() already
# does this same probe-and-fallback, this just makes the startup banner
# reflect reality instead of assuming a GPU is always present.
if [ -z "${POKEMON_TRAIN_DEVICE:-}" ]; then
  if python3 -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    export POKEMON_TRAIN_DEVICE="cuda"
  else
    export POKEMON_TRAIN_DEVICE="cpu"
  fi
fi
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

echo "Using POKEMON_TRAIN_DEVICE=${POKEMON_TRAIN_DEVICE}"
echo "Using POKEMON_VEC_ENV=${POKEMON_VEC_ENV}"
echo "Using POKEMON_SUBPROC_START_METHOD=${POKEMON_SUBPROC_START_METHOD}"

if [ "${POKEMON_TRAIN_DEVICE}" = "cuda" ]; then
  echo "Using HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
  python3 -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print(f'GPU OK: {torch.cuda.get_device_name(0)}')"
else
  echo "No GPU requested/detected -- running on CPU"
fi

exec python3 train.py
