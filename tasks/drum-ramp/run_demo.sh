#!/usr/bin/env bash
# usage: run_demo.sh LABEL MODE MAX_STEPS   (extra -E/-P env via EXTRA; DEMO_MODE knobs via the environment)
# Writes ~/robocurve/eval2/LABEL/ (inspect-robots log + frames), LABEL.csv (trace) and LABEL.log. Refuses to overwrite.
set -u
L=$1; MODE=$2; STEPS=$3
EVAL=$HOME/robocurve/eval2; OUT=$EVAL/$L; TRACE=$EVAL/$L.csv
TASK=$HOME/robocurve/tasks/drum-ramp/drum_ramp.py
for p in "$OUT" "$TRACE"; do [ -e "$p" ] && { echo "REFUSED: $p exists"; exit 6; }; done
source ~/robocurve/IsaacLab-Arena/yam.env
exec 9>>"$HOME/robocurve/eval/.sim.lock"; flock -w 60 9 || { echo "REFUSED: another simulator holds the lock"; exit 7; }
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 DRUM_TRACE=$TRACE DEMO_MODE=$MODE
unset DISPLAY; cd "$(dirname "$TASK")"
TEXT=$(python -c "import drum_ramp as t; print(t.INSTRUCTION)")
echo "start $(date -u +%FT%TZ) task_sha256=$(sha256sum "$TASK" | cut -d' ' -f1) mode=$MODE steps=$STEPS env=$(env | grep -E '^(PUSH_|LIFT_|KNOCK_|SHIFT_|SQUEEZE_|PINCH_)' | tr '\n' ' ')"
timeout --signal=INT --kill-after=60 ${TIMEOUT:-900} inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$TASK -E device=cpu -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 ${EXTRA:-} \
  --instruction "$TEXT" --scorer task_score --max-steps $STEPS --seed 0 --store-frames --no-prompt --log-dir "$OUT"
echo "exit $? end $(date -u +%FT%TZ)"
