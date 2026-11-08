#!/usr/bin/env python
"""Summarise a DRUM_TRACE csv: score, drum path, pusher height relative to the axis, lag, end state. usage: summarize.py TRACE [every]"""
import csv
import sys

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
every = int(sys.argv[2]) if len(sys.argv) > 2 else 20
f = lambda r, k: float(r[k]) if r[k] not in ("", None) else float("nan")
print(f"{'step':>5} {'ts':>4} {'x':>7} {'z':>7} {'spd':>6} {'lift':>4} {'axis':>5} {'deck':>4} {'tipL-ax':>7} {'tipR-ax':>7} {'gapL':>6} {'gapR':>6} {'score':>5}")
for i, r in enumerate(rows):
    if i % every == 0 or i == len(rows) - 1:
        print(f"{r['step']:>5} {r['trial_step']:>4} {f(r,'x'):7.4f} {f(r,'z'):7.4f} {f(r,'speed'):6.3f} {r['lifted']:>4} {f(r,'axis_deg'):5.1f} {r['on_deck']:>4} "
              f"{f(r,'lz')-f(r,'z'):+7.4f} {f(r,'rz')-f(r,'z'):+7.4f} {f(r,'gap_left'):6.4f} {f(r,'gap_right'):6.4f} {r['score']:>5}")
x = np.array([f(r, "x") for r in rows]); z = np.array([f(r, "z") for r in rows])
print("max x", x.max().round(4), "at step", int(x.argmax()), "| max z", z.max().round(4), "| final", rows[-1]["x"], rows[-1]["z"], "score", rows[-1]["score"], "lifted", rows[-1]["lifted"])
