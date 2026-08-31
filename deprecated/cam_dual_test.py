"""Do both cameras still deliver frames when opened together?

The rollout screenshot shows observation.wrist as a solid black panel while
observation.top renders the scene normally. If the wrist feed really is blank
during deployment, the policy is receiving half its input as an empty image --
something it never saw in training -- and any output is meaningless.

Recording worked, so each camera is fine on its own. The suspect is bandwidth:
both share one USB 2.0 bus, and the record log showed
    OpenCVCamera(N) failed to set fourcc=MJPG (actual=, success=False)
so they are running uncompressed. 640x480 YUY2 at 30 fps is ~18 MB/s each;
two of those exceed what one USB 2.0 bus delivers in practice.

This opens them exactly the way lerobot does during rollout and reports the
mean pixel value of each frame over time. A feed that is black reads ~0.

ASCII output only.
"""

import json
import time

import numpy as np

import lerobot_patch  # noqa: F401  (not needed here, but keeps import order uniform

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

m = json.load(open("camera_map.json"))
cams = {}
for name in ("top", "wrist"):
    cfg = OpenCVCameraConfig(index_or_path=int(m[name]), width=640, height=480, fps=30)
    cams[name] = OpenCVCamera(cfg)

for name, c in cams.items():
    c.connect()
    print("connected %-6s index %s" % (name, m[name]))
print()

N = 60
stats = {k: [] for k in cams}
t0 = time.perf_counter()
for i in range(N):
    for name, c in cams.items():
        fr = c.read_latest()
        stats[name].append(float(np.asarray(fr).mean()))
    time.sleep(1 / 30)
dt = time.perf_counter() - t0

for name, c in cams.items():
    c.disconnect()

print("read %d frames from each in %.1f s" % (N, dt))
print()
print("%-8s %8s %8s %8s %8s" % ("cam", "mean", "min", "max", "black?"))
for name, v in stats.items():
    v = np.array(v)
    black = (v < 1.0).mean() * 100
    print("%-8s %8.1f %8.1f %8.1f %7.0f%%" % (name, v.mean(), v.min(), v.max(), black))
print()
print("mean near 0 or a high black%% means that feed is dead when both are open.")
