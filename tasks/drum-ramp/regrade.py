#!/usr/bin/env python3
"""Regrade a drum-ramp trace from the written scoring rules, without the task code.

Written by the same agent that wrote drum_ramp.py, from the task.json text; it shares that lineage and is a check,
not independent proof. It uses only the trace's drum pose, the simulator's speeds, the environment step count, and the arms'
link poses (TRACE.arms.csv) placing the collision-mesh vertices (TRACE.hulls.csv). The saved `lifted`, `on_deck`,
`arm_gap` and `score` columns are compared with afterwards, never used.

usage: regrade.py TRACE.csv [--dt 0.05]
"""
import argparse
import csv
import gzip
import json
import math
import sys

import numpy as np

TOP = 0.7475
R = 0.10
LEN = 0.26
RAMP_X0, RAMP_X1, DECK_H, DECK_X1, STOP_T = 0.34, 0.45, 0.03, 0.66, 0.02
STOP_X0 = DECK_X1 - STOP_T
HALF_W = 0.20
SLOPE = math.atan2(DECK_H, RAMP_X1 - RAMP_X0)
ON_DECK_GAP, PAST_EDGE, ON_DECK_Y, AXIS_DEG = 0.005, 0.05, 0.04, 15.0
CLEAR, LIFT, PARTWAY = 0.03, 0.015, 0.015
REST_SPEED, REST_SPIN, HOLD_S = 0.01, 0.5, 1.0
FIRST_EVAL_STEP = 11


def surface_z(x, y):
    if abs(y) > HALF_W or x < RAMP_X0 or x > DECK_X1:
        return TOP
    if x < RAMP_X1:
        return TOP + (x - RAMP_X0) * math.tan(SLOPE)
    return TOP + DECK_H


def rolling_height(x, y):
    if abs(y) <= HALF_W and RAMP_X0 <= x < RAMP_X1:
        return surface_z(x, y) + R / math.cos(SLOPE)
    return surface_z(x, y) + R


def axis(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)])


def axis_err_deg(q):
    a = axis(q)
    return math.degrees(math.acos(min(1.0, abs(a[1]) / np.linalg.norm(a))))


def cyl_distance(pos, q, pts):
    """Smallest distance from points (n, 3) to the drum's surface, 0 inside."""
    a = axis(q)
    a = a / np.linalg.norm(a)
    d = np.atleast_2d(pts) - pos
    along = d @ a
    radial = np.linalg.norm(d - np.outer(along, a), axis=1)
    return float(np.hypot(np.maximum(radial - R, 0.0), np.maximum(np.abs(along) - LEN / 2, 0.0)).min())


def _open(path):
    """A csv file, or its gzip-compressed copy if only that exists."""
    import os

    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def load_arms(path):
    """{step: [(link, pos, R)]} from TRACE.arms.csv (or .csv.gz)."""
    by_step = {}
    with _open(path) as f:
        for r in csv.DictReader(f):
            pos = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
            rot = np.array([float(r[f"r{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3, 3)
            by_step.setdefault(int(r["step"]), []).append((r["link"], pos, rot))
    return by_step


def load_hulls(path):
    verts = {}
    with _open(path) as f:
        for r in csv.DictReader(f):
            verts.setdefault(r["link"], []).append([float(r["vx"]), float(r["vy"]), float(r["vz"])])
    return {k: np.array(v) for k, v in verts.items()}


def arm_points(arms, hulls, step, n_links):
    links = arms.get(step, [])
    if len(links) != n_links:
        return None
    return np.concatenate([hulls[link] @ rot.T + pos for link, pos, rot in links])


def regrade(path, dt):
    rows = list(csv.DictReader(open(path, newline="")))
    base = path[: -len(".csv")] if path.endswith(".csv") else path
    import glob, os
    folder = os.path.dirname(os.path.abspath(path))
    hull_file = base + ".hulls.csv"
    if not os.path.exists(hull_file) and not os.path.exists(hull_file + ".gz"):
        # the vertices are the same for every run, so one copy per folder is kept
        hull_file = sorted(glob.glob(os.path.join(folder, "*.hulls.csv*")))[0]
    hulls = load_hulls(hull_file)
    n_links = 2 * len(hulls)
    if os.path.exists(base + ".arms.csv") or os.path.exists(base + ".arms.csv.gz"):
        arms = load_arms(base + ".arms.csv")
    else:  # without the arms' poses the clear clause cannot be regraded, so no 1.0 can be awarded
        arms = {}
    f = lambda r, k: float(r[k])
    hold_needed = round(HOLD_S / dt)
    trials, cur = [], None
    for r in rows:
        ts = int(r["trial_step"])
        if cur is None or ts < cur["last_ts"]:
            cur = {"rows": [], "last_ts": -1}
            trials.append(cur)
        cur["rows"].append(r)
        cur["last_ts"] = ts
    results = []
    for tr in trials:
        rs = [r for r in tr["rows"] if int(r["trial_step"]) >= FIRST_EVAL_STEP]
        lifted = False
        start_z = f(rs[0], "z")
        held = 0
        success_at = None
        per_step = []
        mismatches = 0
        for r in rs:
            pos = np.array([f(r, "x"), f(r, "y"), f(r, "z")])
            q = np.array([f(r, "qx"), f(r, "qy"), f(r, "qz"), f(r, "qw")])
            if pos[2] > rolling_height(pos[0], pos[1]) + LIFT:
                lifted = True
            under = pos[2] - R
            on_deck = (
                abs(under - (TOP + DECK_H)) <= ON_DECK_GAP
                and RAMP_X1 + PAST_EDGE <= pos[0] <= STOP_X0
                and abs(pos[1]) <= ON_DECK_Y
                and axis_err_deg(q) <= AXIS_DEG
            )
            still = f(r, "speed") < REST_SPEED and f(r, "spin") < REST_SPIN
            pts = arm_points(arms, hulls, int(r["step"]), n_links)
            gap = 0.0 if pts is None else cyl_distance(pos, q, pts)
            clear = gap >= CLEAR
            complete = not lifted and on_deck and still and clear
            held = held + 1 if complete else 0
            if lifted:
                score = 0.0
            elif on_deck:
                score = 1.0 if still and clear else 0.5
            elif pos[2] - start_z >= PARTWAY:
                score = 0.25
            else:
                score = 0.0
            if success_at is None and held >= hold_needed:
                success_at = int(r["step"])
            per_step.append(score)
            if (abs(score - f(r, "score")) > 1e-9 or int(r["lifted"]) != int(lifted) or int(r["on_deck"]) != int(on_deck)
                    or abs(gap - f(r, "arm_gap")) > 1e-4):
                mismatches += 1
        last = rs[-1]
        final = 1.0 if success_at is not None else per_step[-1]
        results.append(
            {
                "first_step": int(rs[0]["step"]),
                "last_step": int(last["step"]),
                "steps": len(rs),
                "score": final,
                "success_at": success_at,
                "last_step_score": per_step[-1],
                "lifted": lifted,
                "final_pos": [round(float(f(last, k)), 5) for k in ("x", "y", "z")],
                "final_axis_deg": round(axis_err_deg(np.array([f(last, k) for k in ("qx", "qy", "qz", "qw")])), 3),
                "max_height_over_rolling": round(
                    max(f(r, "z") - rolling_height(f(r, "x"), f(r, "y")) for r in rs), 5
                ),
                "final_arm_gap": round(gap, 5),
                "arms_logged": bool(arms),
                "column_mismatches": mismatches,
                "reported_last_score": f(last, "score"),
            }
        )
    return {"trace": path, "dt": dt, "trials": results, "score": results[-1]["score"] if results else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()
    json.dump(regrade(a.trace, a.dt), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
