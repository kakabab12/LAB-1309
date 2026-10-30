#!/usr/bin/env bash
# All poster renders, one after another (run inside a memory-capped systemd unit).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R=paper/poster/render_scenes.py
ACT_DESIGN=v5 $PY $R basic
ACT_OBJECT=cup ACT_DESIGN=cuprule $PY $R object --name cup
ACT_OBJECT=box ACT_DESIGN=v5 $PY $R object --name box
ACT_DESIGN=v5 $PY $R clutter
ACT_DESIGN=v5 $PY $R factory
ACT_DESIGN=v5 $PY $R looks
ACT_DESIGN=v4 PALLET_VIA=far $PY $R pallet
echo "renders: done"
