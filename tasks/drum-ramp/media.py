#!/usr/bin/env python
"""Render a run's stored frames to H.264 MP4s (one per camera) and save first/last frames as PNG.

usage: media.py RUN_DIR OUT_DIR PREFIX   (needs ffmpeg, numpy and PIL; frames are 320x176 RGB .npy at 20 Hz)
"""
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

run, out, prefix = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
ff = os.environ.get("FFMPEG", "ffmpeg")
for cam, short in (("top_cam", "top"), ("left_cam", "left"), ("right_cam", "right")):
    fs = sorted(glob.glob(os.path.join(run, "frames", "*", f"*{cam}_*.npy")))
    idx = [int(f[-10:-4]) for f in fs]
    assert idx == list(range(len(idx))), cam
    mp4 = os.path.join(out, f"{prefix}_{short}.mp4")
    p = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x176", "-r", "20", "-i", "-",
                          "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4],
                         stdin=subprocess.PIPE)
    for f in fs:
        a = np.load(f)
        assert a.shape == (176, 320, 3) and a.dtype == np.uint8, (f, a.shape, a.dtype)
        p.stdin.write(a.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    d = subprocess.run([ff, "-v", "error", "-xerror", "-i", mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    decoded = len(d.stdout) // (320 * 176 * 3)
    print(f"{short}: {len(fs)} frames in, {decoded} decoded, {os.path.getsize(mp4)} bytes, rc {d.returncode}")
    for name, f in (("first", fs[0]), ("last", fs[-1])):
        png = os.path.join(out, f"{prefix}_{short}_{name}.png")
        Image.fromarray(np.load(f)).save(png)
        assert (np.array(Image.open(png)) == np.load(f)).all()
    if short == "top":
        Image.fromarray(np.load(fs[0])).save(os.path.join(out, f"{prefix}_setup.png"))
