"""Which condition kills the wrist feed: being open alone, or sharing the bus?

Three cases, same code path lerobot uses:
  A  wrist alone
  B  both together, default settings (what rollout does today)
  C  both together, DSHOW backend + MJPG

Recording worked with both cameras open, so the hardware is capable of it.
The record log did warn
    failed to set fourcc=MJPG (actual=, success=False)
which means they ran uncompressed: 640x480 YUY2 at 30 fps is ~18 MB/s each,
and two of those do not fit through one USB 2.0 bus. MSMF (the default backend
on Windows) ignores fourcc requests; DSHOW honours them. Case C tests that.

ASCII output only.
"""

import json

import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode

M = json.load(open("camera_map.json"))


def probe(which, n=40, **cfgkw):
    cams = {}
    for name in which:
        kw = dict(index_or_path=int(M[name]), width=640, height=480, fps=30)
        kw.update(cfgkw)
        cams[name] = OpenCVCamera(OpenCVCameraConfig(**kw))
    out = {}
    try:
        for c in cams.values():
            c.connect()
        vals = {k: [] for k in cams}
        for _ in range(n):
            for name, c in cams.items():
                vals[name].append(float(np.asarray(c.read_latest()).mean()))
        for name, v in vals.items():
            v = np.array(v)
            out[name] = (v.mean(), (v < 1.0).mean() * 100)
    finally:
        for c in cams.values():
            try:
                c.disconnect()
            except Exception:
                pass
    return out


def show(label, res):
    print(label)
    for name, (mean, black) in res.items():
        print("    %-6s mean %7.1f   black %3.0f%%   %s"
              % (name, mean, black, "DEAD" if black > 50 else "ok"))
    print()


import inspect
sig = inspect.signature(OpenCVCameraConfig)
print("OpenCVCameraConfig fields:", [p for p in sig.parameters])
print()

show("A) wrist alone", probe(["wrist"]))
show("B) both, defaults (what rollout does now)", probe(["top", "wrist"]))
try:
    show("C) both, DSHOW + MJPG", probe(["top", "wrist"], backend="DSHOW", fourcc="MJPG"))
except TypeError as e:
    print("C) skipped, config does not take those kwargs: %s" % e)
