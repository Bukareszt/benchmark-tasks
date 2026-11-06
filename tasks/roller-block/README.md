# Roll a heavy block on loose rollers

Seen from above, the block is wider both ways than one gripper's 94 mm opening, and it rests on three loose rollers. Pushing rolls it along, but the rollers travel
at half its speed, so it runs off the leading roller unless a roller left behind is moved to the front.

## Requirements

The demo and control runs below were made on 2026-11-06 on one machine only, aarch64 (NVIDIA GB10, driver 580.173.02, Ubuntu 24.04.4).
The environment was installed there on 2026-11-05. None of this has been run on x86.

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

**Adapter revision.** This task file extends the adapter's private scene module at import. Use adapter `0059d02a1a91`. Another revision
may break it, and that has not been tried.

**CPU physics is required: `-E device=cpu`.** With GPU PhysX the loaded block creeps about 150 mm in 3 s with nothing touching it. The
adapter's README says an NVIDIA RTX GPU is still needed, for rendering.

## Installation

```bash
git clone https://github.com/kairosiann/yam-isaac-lab-arena && cd yam-isaac-lab-arena
git checkout 0059d02a1a91bc6834c1d0846022891df0c78730
./setup.sh                       # clones Arena into ARENA_DIR, default ~/IsaacLab-Arena
source ~/IsaacLab-Arena/yam.env  # at the start of every shell
```

Check that `inspect-robots list embodiments` shows `yam_arena` and `inspect-robots list policies` shows `yam_scripted` and `dream`.
For a model run, the Dream key goes in `~/.dreamlabs/credentials` with mode 600, as the adapter's README describes.

The adapter's installer locks Arena for x86_64 and stopped with a platform error on aarch64. On aarch64 this task's environment
also needed these changes in the Arena checkout before re-running `./setup.sh`. Copies are in `review/env-patches/`:

```bash
git -C ~/IsaacLab-Arena apply $PWD/review/env-patches/arena-aarch64-pyproject.patch
git -C ~/IsaacLab-Arena apply $PWD/review/env-patches/arena-asset-root.patch   # the staging asset address returned 404
cp review/env-patches/arena-aarch64-uv.lock ~/IsaacLab-Arena/uv.lock           # earlier notes: drops 9 packages, open3d among them
```

Run these from this folder. The patches and lock match the test machine's Arena checkout. These commands have not been replayed on a clean machine.

## Run and score

Run from this folder. On aarch64 preload libgomp, and always run without a display:

```bash
source ~/IsaacLab-Arena/yam.env
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1   # aarch64 only
unset DISPLAY
export ROLLER_TRACE=$PWD/outputs/demo.csv               # optional, see below
mkdir -p outputs
TEXT=$(python -c 'import roller_block as t; print(t.INSTRUCTION)')
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/roller_block.py -E device=cpu -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$TEXT" --scorer task_score --max-steps 2400 --seed 0 --store-frames --no-prompt --log-dir outputs/demo
```

The run prints `task_score: <score>`. To regrade a trace without the task code (needs numpy):

```bash
python regrade.py outputs/demo.csv --arms outputs/demo.arms.csv --hull outputs/demo.hulls.csv
```

**Seeds.** `--seed 0` with `-E object_jitter=0`. The layout is fixed. Two runs of the demo gave byte-identical traces, so the repeat shows
the run is deterministic on this machine and adds no independent sample.

**Runtime**, measured with Isaac Sim already installed, on this machine. Wall time is start to end in `review/receipts.txt`:

| Run | Steps | Simulated | Wall time |
| --- | --- | --- | --- |
| Demo, twice | 853 | 43.15 s including 10 settling steps | 166 s, 170 s |
| `no_transfer`, `PUSH_M=0.13` | 420 | 21.5 s | 83 s, 81 s |
| `arm_stays` | 861 | 43.55 s | 175 s |

| Dream-1, one trial | 2,400 | 120.5 s including 10 settling steps | 476 s |

A first start on a fresh machine has not been timed either.

For a model, change `--policy`. The layout is fixed, so keep `-E object_jitter=0`.

Set `ROLLER_TRACE=trace.csv` to record the block and roller poses at every step. Beside it, `trace.arms.csv` records every
moving arm link's pose and `trace.hulls.csv` the collision-mesh vertices those poses place, so the `arm_gap` column can be
recomputed without this file. Use the object's local box corners moved by its pose for the bounding boxes (26 x 3 x 3 cm for
a roller). The `trial` column counts environment resets, `trial_step` is the environment's step count and `t_s` that count
times its control period, including the adapter's settling steps before the policy's first step. `slip` is the running slip total
and `score` the score of that step.
`review/traces/` holds the traces of this revision's runs, as `r3_1_<run>.csv` and `dream3.csv`, with the `.arms.csv` and `.hulls.csv`
files of the demo and the Dream-1 trial. `review/receipts.txt` has the task file hash and start and end time of each scripted run, including
pilot runs made before the task file was frozen; the Dream-1 trial's are in `review/traces/dream_ledger_r3.txt`. `review/regrade_r3.json` is
a regrade of the scripted runs from the written scoring text. It was made by an AI agent working from files that the task's authoring
agent had partly written, so it is a check and not independent proof.

## Dream-1 result

One trial on this exact task file (sha256 `9473dfe0024f...`), `dreamlabs/dream-1-beta`, seed 0, 2,400 steps: **task_score 0**, status success, no
errors. 50 requests, 51 attempts, one retried. The block moved 0.42 cm, and it did not leave its rollers. No roller was carried. The arms were
3.81 cm from the nearest object at the last step. `regrade.py` agrees with this score. An earlier revision of the task also scored 0 with Dream-1;
that trial was on a different file, so it is not a repeat of this one.

Scoring tests need no simulator: `python -m pytest test_scoring.py`.

Negative controls: `DEMO_MODE=no_transfer` pushes the block without carrying a roller, `PUSH_M=0.13` stops that push early,
and `DEMO_MODE=arm_stays` makes the full push but leaves the left finger against the block.

Shortcut attempts, played with `yam_scripted` to see what they score: `DEMO_MODE=low_lift` and `hold_up` (two arms pinch and
raise the block; `SQUEEZE_M`, `LIFT_M`, `SHIFT_M`, `PINCH_STYLE`, `CORNER_IN_M`), `preplace` (put rollers in the zone first),
`drag_rollers` and `hold_roller` (`PUSH_M`), `knock` and `tip` (`KNOCK_M`, `KNOCK_S`, `KNOCK_Z`, `PRESS_M`). What each measured is in
`review/shortcuts.json`; `review/shortcut_replay.py` replays a trace under this revision's and the previous revision's rules.

## Limits

- One fixed layout, shown to work in repeated runs of the same deterministic demo; this is not a reliability study.
- The demo uses both arms, but nothing shows one arm could not do the task.
- Arm clearance is geometry: the distance from the arms' collision-mesh vertices (every link but the fixed base) to the
  bounding boxes of the block and rollers. It reads neither contact forces nor gripper opening, so a gripper closed on nothing
  and held away counts as let go. Mesh triangle edges are up to about 10 cm on the gripper and 12.5 cm on link3. Sampling the
  longest faces, a point on a face is up to about 2.7 cm from its nearest vertex on the gripper, 3.8 cm on link3 and 4.9 cm on
  link2, so a contact in the middle of a large mesh face could in theory read as a gap of about that size. In the demo's final hold, sampling those faces every 4 mm gave the same minimum as the vertices, 5.27 cm; this was measured for that one run and is not part of the score.
- The task sets `SUCCESS_HOLD_S = 0` and counts the 1 s hold in `success()` from the environment's control period. Other control
  rates are covered by synthetic tests only; every run here is at 20 Hz.
- A rolled roller's bounding box, as the adapter computes it, is up to about 6 mm larger than the roller. This makes the arm
  check slightly stricter, never looser.
- The arm check and the trial's step count depend on this file changing the adapter's scene module at import. Lighting and
  cameras are the adapter's own.
- Scoring reads poses and velocities at 20 Hz. The slip rule tells rolled from carried, slid or dragged by comparing the block's
  move with the mean move of the rollers under it. No scripted hold in this repository raised the block in the simulator, so that
  rule's handling of a carry is shown by synthetic tests, not by a simulated carry. See `review/shortcuts.json`.
- No camera shows the block's underside against the rollers or the table. The demo's final wrist views take in the block,
  rollers and strip ends from above, each from its own side, but they do not show that contact, so "never touches the table" is
  judged from the simulator's pose only.
- The framing checks were done by AI agents reading the policy's frames, and no human has done this check. In the last audit, an agent could not
  judge 5 cm of arm clearance or table contact from the images, wrongly took the arms in the final frames to be within 5 cm (measured 12.19 cm) and read
  the block as yawed 15-20 degrees (measured 1.27 degrees).
- The 20 cm/s failure rule is in the scoring text but not in the instruction, so a model is not told about it. A fast push can fail even if the block
  rolls cleanly: `DEMO_MODE=knock`, `KNOCK_M=0.06` rolled it 10.6 cm with 0.27 mm of slip, peaked at 28.6 cm/s and scored 0.
- The slip total needs step-by-step positions, so a real trial needs video as well as a tape measure. Underside height needs a side view.
- Regrading from a saved trace reads rounded poses, so its slip runs slightly above the scorer's (demo 2.44 mm against 1.71 mm). Take a replay
  over 2 cm with the `slip` column under 2 cm as rounding.
- No real trial has been run. The block is 3 kg and moves at table height, so the hazard looks low, but that is a judgement and has not been measured.
- Top camera at 320 x 176: the block's top is close to the camera, so the block looks about as long as the strips, and its
  height hides the far ends of the rollers under it.
