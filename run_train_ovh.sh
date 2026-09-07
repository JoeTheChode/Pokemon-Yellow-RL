#!/usr/bin/env bash
set -euo pipefail
cd ~/pokemon-rl
. .venv/bin/activate
export POKEMON_N_ENVS=48
export POKEMON_VEC_ENV=subproc
exec python train.py
