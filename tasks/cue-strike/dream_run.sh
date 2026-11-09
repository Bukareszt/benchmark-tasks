#!/usr/bin/env bash
# One Dream-1 trial on the frozen cue-strike task file. This spends money: it refuses to start without the owner's cap,
#   DREAM_APPROVED_CAP_USD=<usd> bash dream_run.sh
# and refuses a second run (delete nothing to get round it; ask the owner). Start it detached with nohup setsid.
set -u
EVAL=$HOME/robocurve/eval3; OUT=$EVAL/dream1; TRACE=$EVAL/dream1.csv; TASK=$EVAL/task/cue_strike.py
EXPECT=59059c23e3d4671173e0f9617b329c0867b10226a0a81f42cc4ff63dc070373b
EXPECT_TEXT=93a2e33e1ed9bfa2b683ca12b4c3367438de3138b2af60dc6473103f2fe7d20a
LEDGER=$EVAL/dream_ledger.txt; CRED=$HOME/.dreamlabs/credentials
[[ "${DREAM_APPROVED_CAP_USD:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "REFUSED: no spending decision; set DREAM_APPROVED_CAP_USD to the cap the owner approved"; exit 2; }
[ -e "$LEDGER" ] && { echo "REFUSED: ledger exists, a Dream trial was already started"; cat "$LEDGER"; exit 3; }
for p in "$OUT" "$TRACE"; do [ -e "$p" ] && { echo "REFUSED: $p exists"; exit 6; }; done
[ "$(sha256sum "$TASK" | cut -d" " -f1)" = "$EXPECT" ] || { echo "REFUSED: task file changed"; exit 4; }
[ -f "$CRED" ] && [ "$(stat -c %a "$CRED")" = "600" ] || { echo "REFUSED: credentials missing or not mode 600"; exit 5; }
source ~/robocurve/IsaacLab-Arena/yam.env
umask 077
exec 9>>"$HOME/robocurve/eval/.sim.lock"; flock -w 900 9 || { echo "REFUSED: another simulator holds the lock"; exit 7; }
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 CUE_TRACE=$TRACE
unset DISPLAY DEMO_MODE; cd "$(dirname "$TASK")"
TEXT=$(python -c "import cue_strike as t; print(t.INSTRUCTION)")
[ "$(printf %s "$TEXT" | sha256sum | cut -d" " -f1)" = "$EXPECT_TEXT" ] || { echo "REFUSED: instruction is not the one in task.json"; exit 8; }
echo "started $(date -u +%FT%TZ) task=$EXPECT cap_usd=$DREAM_APPROVED_CAP_USD trials=1 max_steps=1200 expected_chunks=25" > "$LEDGER"
timeout --signal=INT --kill-after=60 1200 inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$TASK -E device=cpu -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$TEXT" --scorer task_score --max-steps 1200 --seed 0 --epochs 1 --store-frames --no-prompt --log-dir "$OUT"
code=$?; echo "ended $(date -u +%FT%TZ) exit=$code" >> "$LEDGER"; echo "exit $code"
