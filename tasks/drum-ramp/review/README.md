Evidence for the frozen task file (sha256 in receipts.txt), 2026-11-08, aarch64 GB10, CPU physics.

- receipts.txt: start/end time, task hash, mode and knobs, exit code and task_score of every run.
- traces/f_<run>.csv and dream1.csv: the task's trace (drum pose, simulator speeds, fingertip midpoints, arm gap, lifted,
  on_deck, score per control step); <run>.arms.csv.gz the link poses of every step and f_demo1.hulls.csv.gz the collision-mesh
  vertices they place (the same for every run, so kept once). regrade.py reads the compressed files as they are.
- regrade/f_<run>.json and dream1.json: ../regrade.py on each trace, run on the machine with the uncompressed files.
- dream_ledger.txt: the Dream trial's start, end, task hash and cap.
- media/demo_top.mp4, demo_left.mp4, demo_right.mp4: the demo's three cameras at 320 x 176, 20 fps, H.264; demo_setup.png
  the first top frame. dream_*.mp4 if a Dream-1 trial was run.
- env-patches/: the local Arena changes for aarch64 (see ../README.md, Installation).
