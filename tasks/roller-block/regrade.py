#!/usr/bin/env python3
"""Regrader for the roller-block task, from the task.json spec text.

History: the first version was written from the "instruction", "setup" and "scoring"
fields of task.json only, before the task code was read, according to its author agent.
The revision 3 rules (the narrower height band and the slip rule) were added afterwards by
the agent that wrote the task code. A later agent then revised it against the current spec
text, under an instruction not to read roller_block.py until its numbers were produced.
None of this is proof of independence: the file's lineage touches the task author, and an
agent cannot prove its own independence.

It never uses the saved flags `failed`, `under`, `held` or `score`, nor the saved `arm_gap`
or `low_z`, for scoring. The saved `slip` is used only in the "slip_column" variant. Those
columns are otherwise only compared afterwards, in "column_checks". The saved `speed` and
`ang_speed` are used for scoring because the spec says "using the simulator's velocities"
and they are the only record of them in a trace.

Usage:
    regrade.py TRACE.csv [--arms ARMS.csv] [--hull HULL.csv]
    regrade.py --all TRACE_DIR --arms-dir DIR [--shortcuts shortcuts.json] [--out OUT.json]

By default ARMS is TRACE with `.csv` replaced by `.arms.csv` and HULL by `.hulls.csv`,
beside the trace or in --arms-dir. Without those files the arm clause cannot be regraded;
the output then says so and no 1.0 can be awarded.

Readings of the spec where it leaves a choice. Each has a variant, so the output
shows whether the choice changes the score:
  * half-diagonal: the exact 3D half-diagonal of 12 x 16 x 8 (0.1077 m). Variant: the
    spec's rounded 10.8 cm.
  * underside: the lowest corner of the block box, as specified. Variant: centre z minus 4 cm.
  * speed: the simulator's velocities (`speed`, `ang_speed`), for the 1 cm/s clause and the
    20 cm/s latch. Variant: backward finite differences of the raw pose.
  * start position: the first evaluated step of the trial, environment step 11 at 20 Hz.
    Variant: the nominal y of 0.07 m from the setup.
  * slip: recomputed from the trace's rounded poses (5 decimals). Rounding is not in the
    spec, so this total is inflated relative to one from the simulator's poses; see
    "slip_rounding" in the output. Variant: the saved `slip` column.
  * Bounds are inclusive except where the spec says "under" or "over".
  * A roller centre exactly on the block's box edge counts as inside.
  * Latched failure overrides success: "0 for the rest of the trial".
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

TABLE_Z = 0.7475
BLOCK_HALF = np.array([0.06, 0.08, 0.04])
ROLLER_HALF = np.array([0.13, 0.015, 0.015])
ROLLER_R = 0.015
ROLLER_REST_Z = TABLE_Z + ROLLER_R
ROLLER_LIFT_MAX = 0.005
GOAL_X = (0.115, 0.385)
GOAL_Y = (-0.25, -0.03)
TILT_ZERO_DEG = 5.0
TILT_LATCH_DEG = 10.0
LOW_MIN = 0.02
LOW_MAX = 0.035
SLIP_MAX = 0.02
SPEED_MAX = 0.2
SPEED_LIMIT = 0.01
ARM_CLEAR = 0.05
MIN_ROLLERS = 2
NOMINAL_START_Y = 0.07
HALF_DIAG_3D = float(np.linalg.norm(BLOCK_HALF))
HALF_DIAG_ROUNDED = 0.108
FIRST_EVAL_STEP = 11
POSE_ROUND = 0.5e-5  # the trace's positions have 5 decimals
OBJECTS = ["block", "roller_0", "roller_1", "roller_2"]
CORNERS = np.array(
    [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float
)


def quat_to_R(q):
    """q as x, y, z, w. Normalised, because the trace rounds it."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def world_box(pos, q, half):
    pts = (quat_to_R(q) @ (CORNERS * half).T).T + pos
    return pts.min(axis=0), pts.max(axis=0)


def tilt_deg(q):
    return math.degrees(math.acos(np.clip(quat_to_R(q)[2, 2], -1.0, 1.0)))


def rot_angle(q0, q1):
    """Rotation angle in rad between two x, y, z, w quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))


def load_trace(path):
    """Return a list of steps in file order, each {name: row of floats}."""
    steps, index = [], {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = int(row["step"])
            if key not in index:
                index[key] = len(steps)
                steps.append({"step": key, "objs": {}})
            parsed = {}
            for k, v in row.items():
                if k != "name":
                    parsed[k] = float(v) if v not in ("", None) else float("nan")
            steps[index[key]]["objs"][row["name"]] = parsed
    return steps


def load_arms(path):
    by_step = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_step.setdefault(int(r["step"]), {})[(r["arm"], r["link"])] = (
                np.array([float(r[k]) for k in ("x", "y", "z")]),
                np.array(
                    [float(r["r%d%d" % (i, j)]) for i in range(3) for j in range(3)]
                ).reshape(3, 3),
            )
    return by_step


def load_hull(path):
    v = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            v.setdefault(r["link"], []).append(
                (float(r["vx"]), float(r["vy"]), float(r["vz"]))
            )
    return {k: np.array(x) for k, x in v.items()}


def box_distance(points, lo, hi):
    d = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    return np.sqrt((d * d).sum(axis=1))


def arm_gap(step_links, hull, expected, boxes):
    """Least vertex-to-box distance over all arm points, and any missing links."""
    missing = sorted("%s/%s" % k for k in expected if k not in step_links)
    missing += sorted("%s/%s(no mesh)" % k for k in step_links if k[1] not in hull)
    best = float("inf")
    for key, (t, R) in step_links.items():
        if key[1] not in hull:
            continue
        pts = hull[key[1]] @ R.T + t
        for lo, hi in boxes:
            best = min(best, float(box_distance(pts, lo, hi).min()))
    return best, missing


def split_trials(steps):
    """A new trial starts where the environment's step count falls back."""
    trials, cur, prev = [], [], None
    for s in steps:
        ts = s["objs"]["block"].get("trial_step", s["step"])
        if prev is not None and ts < prev:
            trials.append(cur)
            cur = []
        cur.append(s)
        prev = ts
    if cur:
        trials.append(cur)
    return trials


def control_dt(steps):
    ts = [s["objs"]["block"].get("t_s", float("nan")) for s in steps]
    d = np.diff(np.array(ts))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 0.05


def measure(trial, dt, arms, hull, expected):
    """Per-step geometry for a trial, without any scoring choice applied."""
    out = []
    prev = None
    slip = 0.0
    last_y = None
    for s in trial:
        slip_step = 0.0
        b = s["objs"]["block"]
        pos = np.array([b["x"], b["y"], b["z"]])
        q = np.array([b["qx"], b["qy"], b["qz"], b["qw"]])
        lo, hi = world_box(pos, q, BLOCK_HALF)
        boxes = [(lo, hi)]
        under = 0
        under_moved = []
        for name in OBJECTS[1:]:
            r = s["objs"].get(name)
            if r is None:
                continue
            rp = np.array([r["x"], r["y"], r["z"]])
            rq = np.array([r["qx"], r["qy"], r["qz"], r["qw"]])
            boxes.append(world_box(rp, rq, ROLLER_HALF))
            if (
                lo[0] <= rp[0] <= hi[0]
                and lo[1] <= rp[1] <= hi[1]
                and rp[2] - ROLLER_REST_Z <= ROLLER_LIFT_MAX
            ):
                under += 1
                if last_y is not None:
                    under_moved.append(rp[1] - last_y[name])
        if last_y is not None:
            moved = pos[1] - last_y["block"]
            rollers = sum(under_moved) / len(under_moved) if under_moved else 0.0
            slip_step = min(abs(moved), abs(moved - 2.0 * rollers))
            slip += slip_step
        last_y = {"block": pos[1], **{n: s["objs"][n]["y"] for n in OBJECTS[1:] if n in s["objs"]}}
        ts = b.get("trial_step", s["step"])
        contiguous = prev is not None and ts - prev["ts"] == 1
        if prev is not None and contiguous:
            lin = float(np.linalg.norm(pos - prev["pos"])) / dt
            ang = rot_angle(prev["q"], q) / dt
        else:
            lin, ang = float("nan"), float("nan")
        if arms is None:
            gap, missing = float("nan"), None
        else:
            gap, missing = arm_gap(arms.get(s["step"], {}), hull, expected, boxes)
        out.append(
            {
                "step": s["step"],
                "ts": ts,
                "contiguous": contiguous,
                "pos": pos,
                "q": q,
                "lo": lo,
                "hi": hi,
                "tilt": tilt_deg(q),
                "under": under,
                "slip": slip,
                "slip_step": slip_step,
                "slip_col": b.get("slip", float("nan")),
                "lin_fd": lin,
                "ang_fd": ang,
                "lin_col": b.get("speed", float("nan")),
                "ang_col": b.get("ang_speed", float("nan")),
                "gap": gap,
                "missing": missing,
                "low_corner": float(lo[2]) - TABLE_Z,
                "low_centre": float(pos[2] - BLOCK_HALF[2]) - TABLE_Z,
            }
        )
        prev = out[-1]
    return out


def clauses(m, speed_source, half_diag):
    in_box = (
        m["lo"][0] >= GOAL_X[0]
        and m["hi"][0] <= GOAL_X[1]
        and m["lo"][1] >= GOAL_Y[0]
        and m["hi"][1] <= GOAL_Y[1]
    )
    lin, ang = (
        (m["lin_fd"], m["ang_fd"]) if speed_source == "fd" else (m["lin_col"], m["ang_col"])
    )
    v = lin + ang * half_diag
    return {
        "in_goal": bool(in_box),
        "tilt": m["tilt"] <= TILT_ZERO_DEG,
        "speed": bool(math.isfinite(v) and v < SPEED_LIMIT),
        "rollers": m["under"] >= MIN_ROLLERS,
        "arms": bool(math.isfinite(m["gap"]) and m["gap"] >= ARM_CLEAR and not m["missing"]),
    }, v


def score_trial(ms, need, speed_source="col", half_diag=HALF_DIAG_3D, low_key="low_corner", start_y=None, slip_key="slip"):
    if start_y is None:
        start_y = float(ms[0]["pos"][1])
    hold = max_hold = 0
    per_step_score = []
    latched_at = success_at = None
    latch_cause = None
    last_score, per_step_hold = 0.0, []
    fail_counts = {k: 0 for k in ("in_goal", "tilt", "speed", "rollers", "arms")}
    for m in ms:
        low = m[low_key]
        lin = m["lin_fd"] if speed_source == "fd" else m["lin_col"]
        if latched_at is None:
            causes = [
                name
                for name, hit in (
                    ("underside_low", low < LOW_MIN),
                    ("underside_high", low > LOW_MAX),
                    ("tilt", m["tilt"] > TILT_LATCH_DEG),
                    ("slip", m[slip_key] > SLIP_MAX),
                    ("speed", lin > SPEED_MAX),
                )
                if hit
            ]
            if causes:
                latched_at = m["step"]
                latch_cause = {
                    "clauses": causes,
                    "underside": low,
                    "tilt": m["tilt"],
                    "slip": m[slip_key],
                    "speed": lin,
                }
        c, _ = clauses(m, speed_source, half_diag)
        for k, ok in c.items():
            if not ok:
                fail_counts[k] += 1
        if not m["contiguous"] and m is not ms[0]:
            hold = 0  # the trace skipped a control step, so "consecutive" is not shown
        hold = hold + 1 if latched_at is None and all(c.values()) else 0
        max_hold = max(max_hold, hold)
        per_step_hold.append(hold)
        if latched_at is not None or m["tilt"] > TILT_ZERO_DEG:
            last_score = 0.0
        elif hold >= need:
            last_score, success_at = 1.0, m["step"]
            per_step_score.append(last_score)
            break
        else:
            d = start_y - float(m["pos"][1])
            last_score = 0.5 if d >= 0.15 else 0.25 if d >= 0.06 else 0.0
        per_step_score.append(last_score)
    return {
        "score": last_score,
        "success_at": success_at,
        "latched_at": latched_at,
        "latch_cause": latch_cause,
        "max_hold": max_hold,
        "fail_counts": fail_counts,
        "per_step_hold": per_step_hold,
        "per_step_score": per_step_score,
    }


def finite(a):
    a = np.array(a, dtype=float)
    return a[np.isfinite(a)]


def column_checks(ms, trial, hold_list, half_diag, score_list=()):
    """Compare the saved columns with the recomputed values. Not used for scoring."""
    lin_d = finite([m["lin_fd"] - m["lin_col"] for m in ms])
    ang_d = finite([m["ang_fd"] - m["ang_col"] for m in ms])
    lowz = []
    under_mis, gap_d, held_mis, failed_any = [], [], [], 0
    slip_d, score_mis = [], []
    for i, (m, s) in enumerate(zip(ms, trial)):
        b = s["objs"]["block"]
        lowz.append(b.get("low_z", float("nan")) - (m["low_corner"] + TABLE_Z))
        if int(b.get("under", -1)) != m["under"]:
            under_mis.append(m["step"])
        if math.isfinite(m["gap"]) and "arm_gap" in b:
            gap_d.append(b["arm_gap"] - m["gap"])
        if i < len(hold_list) and "held" in b and int(b["held"]) != hold_list[i]:
            held_mis.append(m["step"])
        failed_any += int(b.get("failed", 0))
        if "slip" in b:
            slip_d.append(b["slip"] - m["slip"])
        if i < len(score_list) and "score" in b and b["score"] != score_list[i]:
            score_mis.append(m["step"])
    rng = lambda a: None if len(a) == 0 else [float(np.min(a)), float(np.max(a))]
    return {
        "fd_minus_col_speed_range": rng(lin_d),
        "fd_minus_col_spin_range": rng(ang_d),
        "low_z_col_minus_corner_range": rng(finite(lowz)),
        "under_mismatch_steps": under_mis[:20],
        "under_mismatch_count": len(under_mis),
        "arm_gap_col_minus_recomputed_range": rng(finite(gap_d)),
        "held_mismatch_steps": held_mis[:20],
        "held_mismatch_count": len(held_mis),
        "failed_col_steps": failed_any,
        "slip_col_minus_recomputed_range": rng(finite(slip_d)),
        "score_mismatch_steps": score_mis[:20],
        "score_mismatch_count": len(score_mis),
    }


def slip_rounding(ms):
    """How far apart the replayed and saved slip totals are, against rounding alone.

    With each position off by at most POSE_ROUND, a step's block move is off by at most
    2 POSE_ROUND and twice the rollers' mean move by at most 4 POSE_ROUND, so a step's slip
    is off by at most 6 POSE_ROUND, plus 1e-6 for the saved column's own 6 decimals.
    """
    bound = 6 * POSE_ROUND + 1e-6
    diffs = []
    for a, b in zip(ms, ms[1:]):
        diffs.append(b["slip_step"] - (b["slip_col"] - a["slip_col"]))
    diffs = np.array(diffs) if diffs else np.zeros(1)
    moved = sum(1 for m in ms if m["slip_step"] > 0)
    return {
        "replay_total": round(ms[-1]["slip"], 6),
        "column_total": ms[-1]["slip_col"],
        "difference": round(ms[-1]["slip"] - ms[-1]["slip_col"], 6),
        "steps_with_replay_slip": moved,
        "per_step_diff_max": round(float(np.abs(diffs).max()), 7),
        "per_step_rounding_bound": bound,
        "steps_over_bound": int((np.abs(diffs) > bound).sum()),
        "total_rounding_bound": round(bound * (len(ms) - 1), 6),
    }


def regrade(trace_path, arms_path=None, hull_path=None, arms_dir=None):
    steps = load_trace(trace_path)
    dt = control_dt(steps)
    need = int(round(1.0 / dt)) + 1
    d = os.path.dirname(os.path.abspath(trace_path))
    stem = os.path.join(arms_dir or d, os.path.basename(trace_path)[: -len(".csv")])
    if arms_path is None:
        arms_path = stem + ".arms.csv"
    if hull_path is None:
        hull_path = stem + ".hulls.csv"
    arms = hull = None
    expected = set()
    notes = []
    if arms_path and os.path.exists(arms_path) and os.path.exists(hull_path):
        arms = load_arms(arms_path)
        hull = load_hull(hull_path)
        arm_names = {k[0] for v in arms.values() for k in v}
        expected = {(a, l) for a in arm_names for l in hull}
        if len(arm_names) != 2:
            notes.append("expected 2 arms, found %s" % sorted(arm_names))
    else:
        notes.append("arm clause not regradable: no arm poses or mesh vertices")

    trials = split_trials(steps)
    results = []
    for trial in trials:
        evaluated = [s for s in trial if s["objs"]["block"].get("trial_step", FIRST_EVAL_STEP) >= FIRST_EVAL_STEP]
        if len(evaluated) != len(trial):
            notes.append("dropped %d steps before environment step %d" % (len(trial) - len(evaluated), FIRST_EVAL_STEP))
        trial = evaluated
        ms = measure(trial, dt, arms, hull, expected)
        base = score_trial(ms, need)
        variants = {
            "fd_speed": score_trial(ms, need, speed_source="fd")["score"],
            "half_diag_10_8cm": score_trial(ms, need, half_diag=HALF_DIAG_ROUNDED)["score"],
            "slip_column": score_trial(ms, need, slip_key="slip_col")["score"],
            "underside_centre": score_trial(ms, need, low_key="low_centre")["score"],
            "start_nominal": score_trial(ms, need, start_y=NOMINAL_START_Y)["score"],
        }
        last = ms[-1]
        c, v = clauses(last, "col", HALF_DIAG_3D)
        gaps = [i for i in range(1, len(ms)) if not ms[i]["contiguous"]]
        results.append(
            {
                "first_step": ms[0]["step"],
                "last_step": last["step"],
                "first_trial_step": ms[0]["ts"],
                "last_trial_step": last["ts"],
                "steps": len(ms),
                "step_gaps": len(gaps),
                "score": base["score"],
                "success_at": base["success_at"],
                "steps_after_success": (
                    None
                    if base["success_at"] is None
                    else last["step"] - base["success_at"]
                ),
                "latched_at": base["latched_at"],
                "latch_cause": base["latch_cause"],
                "max_hold": base["max_hold"],
                "hold_needed": need,
                "variants": variants,
                "start_y": float(ms[0]["pos"][1]),
                "final": {
                    "pos": [round(float(x), 5) for x in last["pos"]],
                    "y_travel": round(float(ms[0]["pos"][1] - last["pos"][1]), 5),
                    "box_x": [round(float(last["lo"][0]), 5), round(float(last["hi"][0]), 5)],
                    "box_y": [round(float(last["lo"][1]), 5), round(float(last["hi"][1]), 5)],
                    "tilt_deg": round(last["tilt"], 4),
                    "motion": round(v, 6),
                    "under": last["under"],
                    "arm_gap": None if not math.isfinite(last["gap"]) else round(last["gap"], 6),
                    "missing_links": last["missing"],
                    "underside": round(last["low_corner"], 5),
                    "slip": round(last["slip"], 6),
                    "clauses": c,
                },
                "extremes": {
                    "max_tilt_deg": round(max(m["tilt"] for m in ms), 4),
                    "min_underside": round(min(m["low_corner"] for m in ms), 5),
                    "max_underside": round(max(m["low_corner"] for m in ms), 5),
                    "min_arm_gap": round(min((m["gap"] for m in ms if math.isfinite(m["gap"])), default=float("nan")), 6),
                },
                "steps_failing_clause": base["fail_counts"],
                "slip_rounding": slip_rounding(ms),
                "column_checks": column_checks(ms, trial, base["per_step_hold"], HALF_DIAG_3D, base["per_step_score"]),
            }
        )
    return {
        "trace": trace_path,
        "dt": dt,
        "hold_needed": need,
        "trials": len(results),
        "score": results[-1]["score"] if results else None,
        "notes": notes,
        "results": results,
    }


def last_saved_score(trace_path):
    """The saved `score` of the last step. Only compared with, never used for grading."""
    last = None
    with open(trace_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["name"] == "block":
                last = row
    return float(last["score"])


def run_all(trace_dir, arms_dir, shortcuts):
    """Regrade every r3_1_*.csv in trace_dir and set each against its reported score."""
    expected = {}
    if shortcuts:
        with open(shortcuts) as f:
            for item in json.load(f)["simulated"]:
                expected[os.path.basename(item["trace"])] = item["score"]
    out = {}
    for name in sorted(os.listdir(trace_dir)):
        if not (name.startswith("r3_1_") and name.endswith(".csv")):
            continue
        path = os.path.join(trace_dir, name)
        r = regrade(path, arms_dir=arms_dir)
        reported = last_saved_score(path)
        r["trace"] = name
        r["reported"] = reported
        r["agrees"] = r["score"] == reported
        if name in expected:
            r["shortcuts_json"] = expected[name]
        out[name[len("r3_1_") : -len(".csv")]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--arms")
    ap.add_argument("--hull")
    ap.add_argument("--all", help="directory of r3_1_*.csv traces")
    ap.add_argument("--arms-dir", help="directory of <name>.arms.csv and <name>.hulls.csv")
    ap.add_argument("--shortcuts")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.all:
        res = run_all(a.all, a.arms_dir, a.shortcuts)
    else:
        res = regrade(a.trace, a.arms, a.hull, a.arms_dir)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    else:
        json.dump(res, sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
