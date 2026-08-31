"""How far is the live camera view from anything the policy saw in training?

WHY
    Everything upstream checks out: the model predicts recorded actions to
    ~1 unit, the 100-step chunk stays within ~2 units, the camera mapping
    matches, the start pose is in distribution, and max_relative_target never
    binds. Yet the arm flails on the robot.

    The one thing that differs between offline and online is the observation
    itself. The dataset was recorded around 23:00 under artificial light; this
    eval ran at midday. With all 30 episodes captured under a single lighting
    condition, a large appearance shift is enough to put every frame out of
    distribution.

    This measures the gap directly rather than arguing about it.

ASCII output only.
"""

import glob

import cv2
import numpy as np


def sig(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32)).astype(np.float64)
    v = g - g.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v), float(g.mean())


def train_sigs(key, n=40):
    files = sorted(glob.glob(f"datasets/so101_v3/videos/observation.images.{key}/**/*.mp4",
                            recursive=True))
    cap = cv2.VideoCapture(files[0])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000
    S, B = [], []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            s, b = sig(fr)
            S.append(s); B.append(b)
    cap.release()
    return np.array(S).reshape(len(S), -1), np.array(B)


def live(index, n=8):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    S, B = [], []
    for k in range(n + 5):
        ok, fr = cap.read()
        if ok and k >= 5:
            s, b = sig(fr)
            S.append(s); B.append(b)
    cap.release()
    return np.array(S).reshape(len(S), -1), np.array(B)


import json
m = json.load(open("camera_map.json"))
print("camera_map: top=%s wrist=%s" % (m["top"], m["wrist"]))
print()
print("%-8s %14s %14s %12s %12s" % ("cam", "train bright", "live bright", "ratio", "best corr"))
for key in ("top", "wrist"):
    TS, TB = train_sigs(key)
    LS, LB = live(int(m[key]))
    # nearest-neighbour correlation of each live frame to any training frame
    best = (LS @ TS.T).max(axis=1)
    # within-training baseline: how similar are training frames to each other
    C = TS @ TS.T
    np.fill_diagonal(C, -1)
    base = C.max(axis=1)
    print("%-8s %14.1f %14.1f %12.2f %12.3f"
          % (key, TB.mean(), LB.mean(), LB.mean() / max(TB.mean(), 1e-6), best.mean()))
    print("%-8s %14s %14s %12s %12.3f"
          % ("", "", "", "(train-train baseline)", base.mean()))
print()
print("Brightness ratio far from 1.0, or best-corr well below the train-train")
print("baseline, means the live view is outside what the policy ever saw.")
