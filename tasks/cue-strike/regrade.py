#!/usr/bin/env python3
"""Regrade a cue-strike trace from the written scoring rules, without the task code.

Written by the same agent that wrote cue_strike.py, from the task.json text; it shares that lineage and is a check,
not independent proof. It uses only the trace's ball and stick poses, the simulator's ball speeds, the environment step
count, and the arms' link poses (TRACE.arms.csv) placing the collision-mesh vertices (TRACE.hulls.csv). The saved
`foul`, `in_goal` and `score` columns are compared with afterwards, never used.

usage: regrade.py TRACE.csv [--dt 0.05]
"""
import argparse
import csv
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

TOP = 0.7475
FLOOR = TOP + 0.004
R = 0.025
STICK_HALF = np.array([0.17, 0.01, 0.01])
LINE_NEAR_X = 0.42 - 0.0075
GOAL_X = (0.70, 0.80)
GOAL_Y = (0.07, 0.21)
GOAL_C = (0.75, 0.14)
RED0 = (0.53, 0.0)
ARM_TOUCH, STICK_TOUCH, ON_MAT = 0.02, 0.003, 0.005
REST, NEAR, MOVED, HOLD_S = 0.01, 0.10, 0.05, 1.0
FIRST_EVAL_STEP = 11
CORNERS = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)


def rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _open(path):
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def load_arms(path):
    by_step = {}
    with _open(path) as f:
        for r in csv.DictReader(f):
            pos = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
            m = np.array([float(r[f"r{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3, 3)
            by_step.setdefault(int(r["step"]), []).append((r["link"], pos, m))
    return by_step


def load_hulls(path):
    v = {}
    with _open(path) as f:
        for r in csv.DictReader(f):
            v.setdefault(r["link"], []).append([float(r["vx"]), float(r["vy"]), float(r["vz"])])
    return {k: np.array(x) for k, x in v.items()}


def regrade(path, dt):
    rows = list(csv.DictReader(open(path, newline="")))
    base = path[: -len(".csv")]
    hull_file = base + ".hulls.csv"
    if not os.path.exists(hull_file) and not os.path.exists(hull_file + ".gz"):
        hull_file = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(path)), "*.hulls.csv*")))[0]
    hulls = load_hulls(hull_file)
    arms = load_arms(base + ".arms.csv") if (os.path.exists(base + ".arms.csv") or os.path.exists(base + ".arms.csv.gz")) else {}
    n_links = 2 * len(hulls)
    f = lambda r, k: float(r[k])
    rs = [r for r in rows if int(r["trial_step"]) >= FIRST_EVAL_STEP]
    foul, foul_step, held, success_at, mismatches, per = None, None, 0, None, 0, []
    red0 = np.array([f(rs[0], "red_x"), f(rs[0], "red_y")])
    for r in rs:
        step = int(r["step"])
        p = {o: np.array([f(r, f"{o}_x"), f(r, f"{o}_y"), f(r, f"{o}_z")]) for o in ("white", "red", "stick")}
        qs = np.array([f(r, f"stick_q{k}") for k in "xyzw"])
        rs_ = rot(qs)
        stick_front = float(((CORNERS * STICK_HALF) @ rs_.T + p["stick"])[:, 0].max())
        local = (p["red"] - p["stick"]) @ rs_
        stick_red = float(np.linalg.norm(np.maximum(np.abs(local) - STICK_HALF, 0.0))) - R
        links = arms.get(step, [])
        pts = np.concatenate([hulls[l] @ m.T + t for l, t, m in links]) if len(links) == n_links else None
        found = []
        if stick_front > LINE_NEAR_X:
            found.append("stick_over_line")
        if pts is None or pts[:, 0].max() > LINE_NEAR_X:
            found.append("arm_over_line")
        for o in ("white", "red"):
            if pts is None or np.linalg.norm(pts - p[o], axis=1).min() - R < ARM_TOUCH:
                found.append(f"arm_touched_{o}")
        if stick_red < STICK_TOUCH:
            found.append("stick_touched_red")
        if found and foul is None:
            foul, foul_step = ",".join(found), step
        x, y, z = p["red"]
        in_goal = GOAL_X[0] + R <= x <= GOAL_X[1] and GOAL_Y[0] <= y <= GOAL_Y[1] and abs(z - (FLOOR + R)) <= ON_MAT
        resting = f(r, "red_speed") < REST and f(r, "red_spin") < 0.5
        if foul:
            score = 0.0
        elif in_goal and resting:
            score = 1.0
        elif math.hypot(x - GOAL_C[0], y - GOAL_C[1]) <= NEAR:
            score = 0.5
        elif np.linalg.norm(p["red"][:2] - red0) >= MOVED:
            score = 0.25
        else:
            score = 0.0
        held = held + 1 if score == 1.0 else 0
        if success_at is None and held >= round(HOLD_S / dt):
            success_at = step
        per.append(score)
        if abs(score - f(r, "score")) > 1e-9 or (r["foul"] or None) != foul or int(r["in_goal"]) != int(in_goal):
            mismatches += 1
    last = rs[-1]
    return {
        "trace": os.path.basename(path),
        "steps": len(rs),
        "score": 1.0 if success_at is not None else per[-1],
        "success_at": success_at,
        "foul": foul,
        "foul_step": foul_step,
        "red_end": [round(f(last, k), 5) for k in ("red_x", "red_y", "red_z")],
        "white_end": [round(f(last, k), 5) for k in ("white_x", "white_y", "white_z")],
        "arms_logged": bool(arms),
        "column_mismatches": mismatches,
        "reported_last_score": f(last, "score"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()
    json.dump(regrade(a.trace, a.dt), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
