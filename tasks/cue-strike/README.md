# Pot a red ball by striking the white ball with a stick

A white and a red ball lie on a felt mat. A yellow line crosses the mat; the red ball and a three-sided goal box are
beyond it, and neither the arms nor the stick may cross the line. The only way to move the red ball is to strike the
white ball with the wooden stick so that it hits the red ball at the point that sends it into the box: the red ball must
go about 32 degrees to the left of the line between the balls, so a straight hit misses.

## Requirements

Runs were made on one machine only, aarch64 (NVIDIA GB10, Ubuntu 24.04.4), with the environment installed there on
2026-11-05 for an earlier task. Nothing here has been run on x86.

| Component | Version |
| --- | --- |
| `yam-isaac-lab-arena` | `0059d02a1a91bc6834c1d0846022891df0c78730`, 0 modified files |
| Isaac Lab-Arena | `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, branch `release/0.3.1`, 3 local changes on aarch64 (see Installation) |
| Isaac Lab submodule | `91a0906f93802eb859192dcb0bca879f891e92d6` |
| Isaac Sim | 6.1.0.0 |
| inspect-robots | 1.0.0 |
| inspect-robots-yam | 0.38.0 |
| inspect-robots-dream | 0.1.0, `b5e936a201cec66c54c297e522c920d98445e339`, with dreamlabs-client 1.0.14 |
| Python, torch, numpy | 3.12.14, 2.11.0+cu130, 2.3.1 |

**Adapter revision.** The task file extends the adapter's private scene module at import (balls with their own material
and damping, a mat with a painted line, bounce-absorbing walls, and the arms' measured points and step count passed to
scoring). Use adapter `0059d02a1a91`; another revision may break it, and that has not been tried.

**CPU physics: `-E device=cpu`.** Every run here used it.

## Installation

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena && cd yam-isaac-lab-arena
git checkout 0059d02a1a91bc6834c1d0846022891df0c78730
./setup.sh                       # clones Arena into ARENA_DIR, default ~/IsaacLab-Arena
source ~/IsaacLab-Arena/yam.env  # at the start of every shell
```

Check that `inspect-robots list embodiments` shows `yam_arena` and `inspect-robots list policies` shows `yam_scripted` and
`dream`. For a model run, the Dream key goes in `~/.dreamlabs/credentials` with mode 600, as the adapter's README describes.

On aarch64 the adapter's installer stopped with a platform error; the Arena checkout needed the three changes kept in
`review/env-patches/` (the same as for our earlier tasks) before re-running `./setup.sh`:

```bash
git -C ~/IsaacLab-Arena apply $PWD/review/env-patches/arena-aarch64-pyproject.patch
git -C ~/IsaacLab-Arena apply $PWD/review/env-patches/arena-asset-root.patch   # the staging asset address returned 404
cp review/env-patches/arena-aarch64-uv.lock ~/IsaacLab-Arena/uv.lock
```

These commands have not been replayed on a clean machine.

## Run and score

Run from this folder. On aarch64 preload libgomp, and always run without a display:

```bash
source ~/IsaacLab-Arena/yam.env
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1   # aarch64 only
unset DISPLAY
export CUE_TRACE=$PWD/outputs/demo.csv                  # optional, see below
mkdir -p outputs
TEXT=$(python -c 'import cue_strike as t; print(t.INSTRUCTION)')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/cue_strike.py -E device=cpu -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$TEXT" --scorer task_score --max-steps 1200 --seed 0 --store-frames --no-prompt --log-dir outputs/demo
```

The run prints `task_score: <score>`. For a model, change `--policy`. The layout is fixed, so keep `-E object_jitter=0`.
`run_demo.sh LABEL MODE STEPS` is the wrapper used for every scripted run here (it writes to `~/robocurve/eval3`).

Scoring tests need no simulator: `python -m pytest test_scoring.py`. To regrade a trace from the written rules without the
task code (needs numpy): `python regrade.py outputs/demo.csv`.

**Seeds.** `--seed 0` with `-E object_jitter=0`. The layout is fixed and the demo is open loop.

**Runtime** on this machine, Isaac Sim already installed (start to end in `review/receipts.txt`): the demo runs 283
control steps (14.2 s simulated) in 35 to 36 s wall time; the scripted controls 22 to 48 s each. A first start on a fresh
machine has not been timed.

**Trace.** With `CUE_TRACE=trace.csv` the task records, at every control step, the poses and simulator speeds of both
balls and the stick, the stick's and arms' foremost x, the arms' gaps to both balls, the stick's gap to the red ball, the
first foul, whether the red ball is in the goal, and the step's score. Beside it, `trace.arms.csv` records every moving arm
link's pose and `trace.hulls.csv` the collision-mesh vertices those poses place, so the arm rules can be recomputed without
this file.

## Results

All runs below are on the frozen task file, sha256 `59059c23e3d4671173e0f9617b329c0867b10226a0a81f42cc4ff63dc070373b`,
adapter `0059d02a1a91` unmodified, `-E device=cpu`, 20 Hz, 320 x 176, seed 0, jitter 0, on 2026-11-09.

**Demo: task_score 1.0.** The right arm picks the stick up 9 cm behind its centre, turns it along the aim line, lowers its
tip to the white ball's centre height and strikes (10 cm of hand travel in 0.25 s). The white ball leaves at 0.54 m/s and
hits the red ball, which leaves at 0.18 m/s, 31 degrees to the left, rolls into the goal box and stops against its left
wall; the episode ends on success at step 282. Run twice: the traces are byte-identical, so the repeat shows the run is
deterministic on this machine and adds no independent sample. The aim includes a correction of +2.2 degrees found by
trial (the held stick turns in the grip during the strike); in a sweep before the freeze, corrections from 1.5 to 3.0
degrees all scored 1.0, and on the frozen file 0 degrees also scored 1.0 and 4.5 degrees scored 0.25.

**Dream-1: task_score 0.0.** One trial on this exact task file, `dreamlabs/dream-1-beta`, seed 0, 1,200 steps (the step
limit), status completed, 2 min 6 s wall time (`review/dream_ledger.txt`); 25 requests of 48 action rows, 25 attempts, no
retries, one session. The right arm took the stick near one end at about step 300, raised it nearly upright (its centre
up to 31 cm above the mat) and moved it about; at step 358 the stick passed over the yellow line, a foul, so the trial
scored 0 from then on. Neither ball was touched or moved; the closest any arm point came to a ball was 5.1 cm. Its
three videos are `review/media/dream_*.mp4`, its trace `review/traces/dream1.csv`, and `regrade.py` agrees with the 0
(`review/regrade/dream1.json`). Without the line rule the trial would also have scored 0: the balls never moved.

`review/regrade/*.json` regrades every run from the written rules using only the trace, the simulator's ball speeds and
the logged link poses placing the collision-mesh vertices (`regrade.py`); it agrees with the task's score on every run and
on every step's `foul`, `in_goal` and `score` columns (0 mismatches). It was written by the same agent that wrote the
task, so it is a check, not independent proof.

## Controls and shortcut attempts

`DEMO_MODE` selects a scripted variant; knobs are environment variables (`review/batch_knobs.txt` lists each run's).

| Run | What it does | task_score | What happened |
| --- | --- | --- | --- |
| `full` (demo), twice | the strike above | 1.0 | red ball at rest in the goal at x 0.766, y 0.185 |
| `idle` | the right arm moves aside; nothing is struck | 0 | nothing moves |
| `aim_off` | the same strike aimed straight at the red ball's centre | 0.25 | the red ball rolls nearly straight on and stops by the far rail at y 0.00, 7 cm right of the goal's floor |
| `full`, `AIM_DEG=4.5` | the strike, aim 2.3 degrees off the demo's | 0.25 | the red ball stops 5 cm short of the goal's mouth, 10.7 cm from the goal's centre (0.5 needs 10 cm) |
| `full`, `AIM_DEG=0` | the strike, aim 2.2 degrees off the demo's | 1.0 | the red ball still enters the goal |
| `hand_push` | the closed fingers strike the white ball instead of the stick | 0 | foul (arm within 2 cm of the white ball) at step 71. Without the rule it would have scored 0.25: the white ball left at 0.71 m/s and sent the red one 21 cm, past the goal's right side |
| `stick_push` | the stick pushes the white ball slowly all the way to the red one | 0 | foul (stick over the line) at step 271; the white ball had drifted off the stick's tip and never reached the red one |
| `reach_red`, left arm and right arm | an arm reaches over the line to push the red ball into the goal | 0 | foul (arm over the line) at step 15 or 16; with a downward grip neither arm reached the red ball (closest 3 cm) |

The 0.5 tier (red ball within 10 cm of the goal's centre but not in it) did not occur in these runs; it is covered by
`test_scoring.py` (22 tests). No scripted run tried throwing the stick or striking with it held beyond the line between
two control steps; the rules are checked at 20 Hz only.

## Limits

- One fixed layout and an open-loop demo with an aim correction found by trial: a feasibility demonstration, not a
  reliability study.
- One arm suffices: the demo uses the right arm only. Nothing in the task requires two.
- The felt's rolling resistance is approximated by linear and angular damping of 0.3/s on the balls; ball-to-ball friction
  is 0.1 and restitution 0.9; the walls, rails and mat absorb bounces (restitution 0). With PhysX's default sleeping a
  rolling ball stopped dead at 0.16 m/s in a pilot run, so sleeping is disabled on the balls.
- The stick is a square wooden rod, not a tapered cue: a round stick rolled on the mat before it was picked up and turned
  in the grip.
- The arm rules are geometry: distances from the arms' collision-mesh vertices (every link but the fixed base). A point on
  a mesh face between vertices is not measured, which is why the touch distance is 2 cm rather than contact. Contact
  forces are not read. The stick's rules use its exact box shape.
- Everything is checked at the 20 Hz control steps; what happens between them is not seen.
- What the cameras show (AI audit of the policy's 320 x 176 frames, no human check): the balls, stick, line and goal are
  identifiable in the first top frame, and whether the red ball ends in the goal can be read from the top camera and the
  left wrist camera. The audit's points acted on before the freeze: "behind the line" was unclear (now "on your side of
  the yellow line"), "inside" an open goal box was unclear (its floor is now painted light blue), the light stick blended
  with the table (now dark-stained). Not fixed: the red ball is about 6 pixels across in the top view, so the exact
  contact point cannot be read from it, only the direction to the goal; neither a 2 cm gap nor a raised arm's position
  against the line can be judged from the images; the right wrist camera sees only the stick and table while holding it.
- No real trial has been run. 160 g balls rolling under 1 m/s at table height are a low hazard; that is a judgement, not a
  measurement.
- All runs on aarch64 only, with the local Arena changes listed under Installation; not compared with x86.
