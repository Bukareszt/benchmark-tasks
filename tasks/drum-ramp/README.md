# Roll a drum up a ramp onto a deck

A 20 cm plastic drum lies on the table in front of the arms, too wide for either gripper. A ramp rises 3 cm to a deck
with a stop bar along its far edge. The drum has to be rolled up the ramp and onto the deck until it rests against the
stop; released on the ramp it rolls back, and pushed unevenly it skews off the side.

## Requirements

Runs were made on one machine only, aarch64 (NVIDIA GB10, driver 580.173.02, Ubuntu 24.04.4), with the environment
installed there on 2026-11-05 for an earlier task. Nothing here has been run on x86.

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

**Adapter revision.** The task file extends the adapter's private scene module at import (a lying cylinder with its own
friction, a pitched fixed box, and the fingertip positions and step count passed to scoring). Use adapter `0059d02a1a91`;
another revision may break it, and that has not been tried.

**CPU physics: `-E device=cpu`.** Every run here used it; with GPU PhysX a loaded body on the same stack crept with
nothing touching it in an earlier task. The adapter's README says an NVIDIA RTX GPU is still needed, for rendering.

## Installation

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena && cd yam-isaac-lab-arena
git checkout 0059d02a1a91bc6834c1d0846022891df0c78730
./setup.sh                       # clones Arena into ARENA_DIR, default ~/IsaacLab-Arena
source ~/IsaacLab-Arena/yam.env  # at the start of every shell
```

Check that `inspect-robots list embodiments` shows `yam_arena` and `inspect-robots list policies` shows `yam_scripted` and
`dream`. For a model run, the Dream key goes in `~/.dreamlabs/credentials` with mode 600, as the adapter's README describes.

The adapter's installer locks Arena for x86_64 and stopped with a platform error on aarch64. On aarch64 the environment
also needed these changes in the Arena checkout before re-running `./setup.sh`. Copies are in `review/env-patches/`:

```bash
git -C ~/IsaacLab-Arena apply $PWD/review/env-patches/arena-aarch64-pyproject.patch
git -C ~/IsaacLab-Arena apply $PWD/review/env-patches/arena-asset-root.patch   # the staging asset address returned 404
cp review/env-patches/arena-aarch64-uv.lock ~/IsaacLab-Arena/uv.lock           # drops 9 packages, open3d among them
```

These commands have not been replayed on a clean machine.

## Run and score

Run from this folder. On aarch64 preload libgomp, and always run without a display:

```bash
source ~/IsaacLab-Arena/yam.env
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1   # aarch64 only
unset DISPLAY
export DRUM_TRACE=$PWD/outputs/demo.csv                 # optional, see below
mkdir -p outputs
TEXT=$(python -c 'import drum_ramp as t; print(t.INSTRUCTION)')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/drum_ramp.py -E device=cpu -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$TEXT" --scorer task_score --max-steps 2400 --seed 0 --store-frames --no-prompt --log-dir outputs/demo
```

The run prints `task_score: <score>`. For a model, change `--policy`. The layout is fixed, so keep `-E object_jitter=0`.

Scoring tests need no simulator: `python -m pytest test_scoring.py`. To regrade a trace from the written rules without the
task code (needs numpy): `python regrade.py outputs/demo.csv`.

**Seeds.** `--seed 0` with `-E object_jitter=0`. The layout is fixed and the demo is open loop.

**Runtime** on this machine:

| Run | Steps (control, after 10 settling steps) | Simulated | Wall time |
| --- | --- | --- | --- |
| Demo, twice | 470 (ends on success) | 24.0 s | 54 s, 53 s |
| `arm_stays`, `PUSH_TO=0.60` | 545 (policy stops) | 27.8 s | 61 s |
| `one_arm`, `one_end` | 471 | 24.0 s | 52 s each |
| `knock` | 207, 204 | about 10 s | 30 s, 29 s |
| `lift` | 450 | 22.5 s | 63 s, 65 s |

Wall time is start to end in `review/receipts.txt`, Isaac Sim already installed. A first start on a fresh machine has not been timed.

**Trace.** With `DRUM_TRACE=trace.csv` the task records, at every control step, the drum's pose and the simulator's speed
and spin, each gripper's fingertip midpoint, the environment's step count (`trial_step`, including the adapter's settling
steps), the lifted latch, the on-deck test and the step's score. Beside it, `trace.arms.csv` records every moving arm link's
pose and `trace.hulls.csv` the collision-mesh vertices those poses place, so the arm gap can be recomputed without this file.
`review/traces/` holds the traces of this revision's runs (`f_<run>.csv`, `dream1.csv`).

## Results

All runs below are on the frozen task file, sha256 `6ea0933566e4...` (full hash in `review/receipts.txt`), adapter
`0059d02a1a91` unmodified, `-E device=cpu`, 20 Hz, 320 x 176, seed 0, jitter 0, on 2026-11-08.

**Demo: task_score 1.0** in 470 control steps; the episode ended on success at step 469 (the drum still against the stop,
its centre at x 0.568, and the arms clear for 1 s). Run twice: the two traces are byte-identical, so the repeat shows the run
is deterministic on this machine and adds no independent sample. The push itself: both hands reach the drum's rear at about
step 90, roll it to the ramp foot by step 180 and up the ramp by step 290 at 2 to 3 cm/s; the drum leaves the ramp at about
3 cm/s and coasts the last 12 cm to the stop on its own while the arms back off and lift away. The drum's centre never
rose more than 3.6 mm above its rolling height (the ramp's foot edge accounts for that).

**Dream-1: task_score 0.0.** One trial on this exact task file, `dreamlabs/dream-1-beta`, seed 0, 2,400 steps (the step limit),
status completed, no errors, 237 s wall time; 50 requests of 48 action rows, 50 attempts, no retries, no rows beyond joint limits, one session, mean round trip 0.63 s. The drum's centre stayed within x 0.214 to 0.235 (it started at 0.230) and never left
the table; an arm touched the drum on 969 of the 2,400 control steps, mostly the right hand resting against it, but no push moved
it toward the ramp. `regrade.py` agrees with the 0. The trial's three videos are `review/media/dream_*.mp4`, its trace
`review/traces/dream1.csv` and its ledger `review/dream_ledger.txt`. The cause of the zero is not established.

`review/regrade/*.json` regrades every run from the written rules using only the trace, the simulator's speeds and the
logged link poses placing the collision-mesh vertices (`regrade.py`; the per-step link poses are kept gzip-compressed for
every run, and the vertex file, the same for every run, once as `f_demo1.hulls.csv.gz`); it agrees with the task's score on every run and on
every step's `lifted`, `on_deck`, `arm_gap` and `score` columns (0 mismatches). It was written by the same agent that wrote
the task, so it is a check, not independent proof.

## Negative controls and shortcut attempts

`DEMO_MODE` selects a scripted variant; knobs are environment variables, recorded at the top of each run's log.

| Mode | What it does | Expected | Why it is run |
| --- | --- | --- | --- |
| `full` (default) | Both arms push the drum from behind, one near each end, with the gripper body's flat face at the drum's axis height and the fingertips run along the surface; up the ramp at 2 cm/s, then onto the deck, then back off and lift away | 1.0 | the demonstration |
| `arm_stays`, `PUSH_TO=0.60` | The same push, commanded past the stop, so the hands keep pressing the drum against it | 0.5 | a drum on the deck but still held must not score 1.0 |
| `arm_stays`, `PUSH_TO=0.40` | The push stops with the drum held partway up the ramp | 0.25 | partial credit for height gained, nothing more |
| `one_arm` | The left arm alone pushes at the drum's middle | reported as found | does one arm suffice? |
| `one_end` | The left arm alone pushes 4 cm from the drum's left end | reported as found | does an uneven push skew the drum off the ramp? |
| `knock`, `KNOCK_V=0.4` (default) and `KNOCK_V=0.8 KNOCK_M=0.12` | Both arms strike the drum from behind at that speed over that stroke and withdraw | 0 unless momentum carries it onto the deck | a shortcut attempt: can it be thrown up the ramp? |
| `lift`, `SQUEEZE_M=0.03` (default) and `SQUEEZE_M=0.08 LIFT_M=0.06` | Both arms squeeze the drum's end faces with the gripper bodies, raise it and carry it toward the deck | 0 (lifted rule, or the drum slips) | a shortcut attempt against "roll": carry it instead |

Results on the frozen file (`review/receipts.txt`, traces in `review/traces/`, regrades in `review/regrade/`):

| Run | task_score | What happened |
| --- | --- | --- |
| `arm_stays`, `PUSH_TO=0.60` | 0.5 | drum against the stop, hands pressing it: arm gap 0, never clear |
| `arm_stays`, `PUSH_TO=0.36`, `0.38`, `0.40` | 1.0 | the hands stop at the ramp's top, but the drum coasts onto the deck to x 0.515 and rests there, clear of the hands: a fair 1.0. The 0.25 tier was reached on the previous revision (identical scoring code; the drum then stopped at the deck's edge, held) and is covered by `test_scoring.py` |
| `one_arm` | 1.0 | one arm at the middle suffices; the drum ends 7.3 degrees skewed, within the 15 degree rule |
| `one_end` | 1.0 | one arm 4 cm from one end also suffices in simulation; 2.7 degrees skewed |
| `knock`, 0.4 m/s | 0 | the drum reaches x 0.42 on the ramp and rolls back to the table |
| `knock`, 0.8 m/s | 0 | it reaches the stop at speed (x 0.577, 8 mm above rolling height), bounces and rolls back to the table |
| `lift`, squeeze 3 cm | 0 | the slick drum is dragged 4 cm and turned 7 degrees, never leaves the table |
| `lift`, squeeze 8 cm | 0 | raised at most 1.0 cm, dragged onto the ramp's foot, dropped, rolled back behind the arms |

What the shortcut attempts did not show: no scripted hold raised the drum more than 1 cm, so the lifted rule's handling of
a real carry rests on synthetic tests. A fast enough strike might in principle carry the drum over the stop; at 0.8 m/s
(about the arm's limit for this motion) it bounced back.

## Limits

- One fixed layout and an open-loop demo; two demo runs gave byte-identical traces. This is a feasibility demonstration, not a
  reliability study.
- One arm suffices in simulation: `one_arm` and `one_end` both reached the stop (see above), so the task does not require
  two arms. The demo uses both because an uneven push skews a real drum.
- The drum's friction is set to 0.25 static / 0.20 dynamic against everything (PhysX combine "min"), below measured
  plastic-on-rubber or plastic-on-steel values. With the adapter's 1.2 the hand was dragged up over the drum in every pilot
  run: the rear of a rolling drum moves upwards, and the arm's joint stiffness lets the hand follow. This is disclosed in
  the fidelity note; a real drum would be harder to push with this hand than the simulated one.
- The drum is a convex-mesh cylinder with no rolling resistance: it coasts across the deck to the stop after leaving the
  ramp (about 20 cm on its own in the demo), and it rolls back down the ramp the moment the push stops.
- Arm clearance is geometry: the distance from the arms' collision-mesh vertices (every link but the fixed base) to the
  drum's surface. Points on a mesh face between vertices are not measured, so a contact in the middle of a large face
  could in theory read as a gap of a few centimetres; the body face that pushes the drum has vertices at its edges, and
  the `arm_stays` control (hands pressing the drum) reads a gap of 0 as intended. Contact forces are not read.
- The lifted rule compares the drum's centre with its rolling height on the surface under its centre, so a drum carried
  within 1.5 cm of that height is not caught. No scripted run lifted the drum more than 1.1 cm (the slick drum slips out
  of a squeeze), so the rule's handling of a real carry rests on synthetic tests, not on a simulated carry.
- The scorer reads poses and the simulator's velocities at 20 Hz; "at any time" means at any control step.
- No camera shows the drum's underside against the ramp, so rolled versus carried is judged from the simulator's pose; the
  stripe along the drum shows rotation when it comes round. The wrist cameras see little during the push (a dark close-up
  of the drum or ramp). No human has checked the frames: the framing checks were done by an AI agent reading the policy's
  images, which found the ramp and deck hard to tell apart when both were grey (the ramp is blue since), the stop hidden
  behind the drum in the final top frame (fixtures widened to 50 cm since) and rolling not visible (stripe added since).
- At the end of a trial the episode stops as soon as the drum has been still and the arms clear for 1 s, so the demo's
  videos end with the arms raised above the ramp rather than parked.
- No real trial has been run. A 2 kg drum at table height is a low hazard, but that is a judgement, not a measurement.
- All runs on aarch64 only, with the local Arena changes listed under Installation; not compared with x86.
