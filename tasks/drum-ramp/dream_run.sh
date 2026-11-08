#!/usr/bin/env bash
# One Dream-1 trial on the frozen drum-ramp task file. This spends money: it refuses to start without the owner's cap,
#   DREAM_APPROVED_CAP_USD=<usd> bash dream_run.sh
# and refuses a second run (delete nothing to get round it; ask the owner). Start it detached with nohup or setsid.
set -u
EVAL=$HOME/robocurve/eval2; OUT=$EVAL/dream1; TRACE=$EVAL/dream1.csv; TASK=$HOME/robocurve/tasks/drum-ramp/drum_ramp.py
EXPECT=6ea0933566e44d98c1e6b393f848afdf6d7092384c7c37f6066cbad77df18e55
EXPECT_TEXT=7dff29c93877ec646131d4f337366d81a63ac194c531e761f2278c6ac4c16451
LEDGER=$EVAL/dream_ledger.txt; CRED=$HOME/.dreamlabs/credentials
[[ "${DREAM_APPROVED_CAP_USD:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "REFUSED: no spending decision; set DREAM_APPROVED_CAP_USD to the cap the owner approved"; exit 2; }
[ -e "$LEDGER" ] && { echo "REFUSED: ledger exists, a Dream trial was already started"; cat "$LEDGER"; exit 3; }
for p in "$OUT" "$TRACE"; do [ -e "$p" ] && { echo "REFUSED: $p exists"; exit 6; }; done
[ "$(sha256sum "$TASK" | cut -d" " -f1)" = "$EXPECT" ] || { echo "REFUSED: task file changed"; exit 4; }
[ -f "$CRED" ] && [ "$(stat -c %a "$CRED")" = "600" ] && grep -q "^DREAM_API_KEY=dl_" "$CRED" || { echo "REFUSED: credentials missing, wrong format or not mode 600"; exit 5; }
source ~/robocurve/IsaacLab-Arena/yam.env
umask 077
exec 9>>"$HOME/robocurve/eval/.sim.lock"; flock -n 9 || { echo "REFUSED: another simulator holds the lock"; exit 7; }
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 DRUM_TRACE=$TRACE
unset DISPLAY DEMO_MODE PUSH_TO; cd "$(dirname "$TASK")"
TEXT=$(python -c "import drum_ramp as t; print(t.INSTRUCTION)")
[ "$(printf %s "$TEXT" | sha256sum | cut -d" " -f1)" = "$EXPECT_TEXT" ] || { echo "REFUSED: instruction is not the one in task.json"; exit 8; }
echo "started $(date -u +%FT%TZ) task=$EXPECT cap_usd=$DREAM_APPROVED_CAP_USD trials=1 max_steps=2400 expected_chunks=50" > "$LEDGER"
timeout --signal=INT --kill-after=60 1800 inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$TASK -E device=cpu -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$TEXT" --scorer task_score --max-steps 2400 --seed 0 --epochs 1 --store-frames --no-prompt --log-dir "$OUT"
code=$?; echo "ended $(date -u +%FT%TZ) exit=$code" >> "$LEDGER"; echo "exit $code"
