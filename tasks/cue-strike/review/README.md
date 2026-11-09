Evidence for the frozen task file (sha256 in receipts.txt), 2026-11-09, aarch64 GB10, CPU physics.

- receipts.txt: start/end time, task hash, mode, exit code and task_score of every scripted run; batch_knobs.txt the
  environment knobs each run was given (AIM_DEG, REACH_ARM).
- traces/f_<run>.csv and dream1.csv: the task's trace (ball and stick poses and speeds, stick and arm foremost x, arm gaps
  to the balls, stick gap to the red ball, first foul, in_goal, score per control step); <run>.arms.csv.gz the link poses
  of every step and f_demo1.hulls.csv.gz the collision-mesh vertices they place (the same for every run, kept once).
  regrade.py reads the compressed files as they are.
- regrade/f_<run>.json and dream1.json: ../regrade.py on each trace.
- dream_ledger.txt and dream1_summary.txt: the Dream trial's start, end, task hash, cap, status and score.
- media/demo_*.mp4 and dream_*.mp4: the three cameras at 320 x 176, 20 fps, H.264; *_setup.png the first top frame,
  *_first/_last.png the first and last frame of each camera.
- env-patches/: the local Arena changes for aarch64 (see ../README.md, Installation).
