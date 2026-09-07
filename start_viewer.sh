#!/usr/bin/env bash
set -euo pipefail
cd ~/pokemon-rl
# Refresh the objective chain that the gate card's "Clears when ..." line is
# built from, so a train.py quest edit only needs a viewer restart to show up.
# Needs the trainer venv (it imports train.py). Never fatal: without it the
# card falls back to naming the objective, which is what it did before.
./.venv/bin/python export_quest_chain.py   || echo 'quest chain export failed; objective card will show names only' >&2
exec python3 map_viewer_server.py
