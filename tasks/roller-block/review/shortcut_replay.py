#!/usr/bin/env python3
"""Replay a roller-block trace under the revision 2 rules (underside 2-4.5 cm, no slip rule) and the revision 3 rules
(underside 2-3.5 cm, slip at most 2 cm, centre speed at most 20 cm/s), and print what each would have scored, with the block's motion.

    shortcut_replay.py TRACE.csv [...]

Reads poses and the saved arm_gap column from the trace, so the arm clause is not recomputed here (regrade.py does that).
This is an offline replay from saved poses (4-5 decimals). The score under the rules a run was made with is in its receipt.
"""
import csv
import json
import sys

import numpy as np

TOP = 0.7475
HALF = np.array([0.06, 0.08, 0.04])
CORNERS = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * HALF
ROLLERS = ("roller_0", "roller_1", "roller_2")
RULES = {"r2_3": dict(lifted=0.045, slip=None, speed=None), "r3": dict(lifted=0.035, slip=0.02, speed=0.2)}
HOLD = 21


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load(path):
    steps = {}
    for r in csv.DictReader(open(path)):
        steps.setdefault(int(r["step"]), {})[r["name"]] = r
    return [steps[k] for k in sorted(steps)]


def replay(steps, lifted, slip_max, speed_max):
    failed, held, slip, last, score, done, why = False, 0, 0.0, None, 0.0, False, None
    start = float(steps[0]["block"]["y"])
    for i, s in enumerate(steps):
        b = s["block"]
        pos = np.array([float(b[k]) for k in "xyz"])
        q = [float(b[k]) for k in ("qx", "qy", "qz", "qw")]
        R = rot(q)
        pts = CORNERS @ R.T + pos
        low, high = pts.min(0), pts.max(0)
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[0] ** 2 + q[1] ** 2), -1, 1))))
        under = [n for n in ROLLERS if low[1] <= float(s[n]["y"]) <= high[1] and low[0] <= float(s[n]["x"]) <= high[0] and float(s[n]["z"]) <= TOP + 0.02]
        ys = {n: float(s[n]["y"]) for n in ("block", *ROLLERS)}
        if last is not None:
            moved = ys["block"] - last["block"]
            rollers = sum(ys[n] - last[n] for n in under) / len(under) if under else 0.0
            slip += min(abs(moved), abs(moved - 2 * rollers))
        last = ys
        h = low[2] - TOP
        cause = "underside below 2 cm" if h < 0.02 else f"underside over {lifted * 100:g} cm" if h > lifted else "tilt over 10 deg" if tilt > 10 else None
        if cause is None and slip_max is not None and slip > slip_max:
            cause = "slip"
        if cause is None and speed_max is not None and float(b["speed"]) > speed_max:
            cause = "speed"
        if cause and not failed:
            failed, why = True, f"{cause} at step {i}"
        speed = float(b["speed"]) + float(b["ang_speed"]) * 0.1077
        ok = (not failed and low[1] >= -0.25 and high[1] <= -0.03 and low[0] >= 0.115 and high[0] <= 0.385 and tilt <= 5
              and len(under) >= 2 and speed < 0.01 and float(b["arm_gap"]) >= 0.05)
        held = held + 1 if ok else 0
        done = held >= HOLD
        travel = start - pos[1]
        score = 0.0 if failed or tilt > 5 else 1.0 if done else 0.5 if travel >= 0.15 else 0.25 if travel >= 0.06 else 0.0
        if done:
            break
    return {"score": score, "latched": why, "slip_m": round(slip, 5), "steps": i + 1}


def summary(path):
    steps = load(path)
    blocks = [s["block"] for s in steps]
    h = np.array([float(b["low_z"]) - TOP for b in blocks]) * 100
    y = np.array([float(b["y"]) for b in blocks])
    out = {
        "steps": len(steps),
        "underside_cm_min_max": [round(float(h.min()), 3), round(float(h.max()), 3)],
        "travel_final_cm": round(float(y[0] - y[-1]) * 100, 2),
        "travel_max_cm": round(float(y[0] - y.min()) * 100, 2),
        "block_speed_max_m_s": max(float(b["speed"]) for b in blocks),
        "final_under": int(blocks[-1]["under"]),
        "final_arm_gap_cm": round(float(blocks[-1]["arm_gap"]) * 100, 2),
        "final_rollers_y": [float(steps[-1][n]["y"]) for n in ROLLERS],
        "rollers_travel_cm": [round((float(steps[0][n]["y"]) - float(steps[-1][n]["y"])) * 100, 2) for n in ROLLERS],
    }
    for name, r in RULES.items():
        out[f"replay_{name}"] = replay(steps, r["lifted"], r["slip"], r["speed"])
    if "score" in blocks[-1]:
        out["trace_slip_m"] = float(blocks[-1]["slip"])
        out["trace_score"] = float(blocks[-1]["score"])
    return out


if __name__ == "__main__":
    print(json.dumps({p: summary(p) for p in sys.argv[1:]}, indent=1))
