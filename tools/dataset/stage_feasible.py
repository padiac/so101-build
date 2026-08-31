"""Could stage 1 be computed instead of learned?

The idea is to split the task: locate the block, move above it, then grasp. The
locating step only needs a mapping from the block's pixel position to the arm
pose that reaches it -- 2 inputs, 6 outputs, 75 examples. If that mapping is
accurate enough, stage 1 becomes arithmetic and only the fine grasp has to be
learned.

Accuracy is measured leave-one-out: fit on 74 episodes, predict the held-out
one, repeat. That is the honest number, not the fit residual.

The detector was checked by eye on all 75 opening frames and found the block
every time, so its inputs can be trusted here.

ASCII output only.
"""

import glob

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT, FPS, GRIP = "datasets/so101_v3", 30, 5
J = ["sh_pan", "sh_lift", "elbow", "wr_flex", "wr_roll", "grip"]

ds = LeRobotDataset("local/v2", root=ROOT)
edf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))
ddf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))]).sort_values(["episode_index", "frame_index"])
act = np.stack(ddf["action"].to_numpy())


def block_xy(i):
    a = (ds[i]["observation.images.top"].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    hsv = cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, (18, 90, 90), (38, 255, 255)),
                         cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return None
    x, y, w, h = cv2.boundingRect(max(c, key=cv2.contourArea))
    return np.array([x + w / 2.0, y + h / 2.0])


def grasp_pose(seg):
    g = seg[:, GRIP]
    o = int(np.argmax(g))
    thr = g.min() + 0.30 * (g.max() - g.min())
    hold = int(0.7 * FPS)
    k = next((i for i in range(o, len(g) - hold) if (g[i:i + hold] < thr).all()), None)
    if k is None:
        k = int(o + np.argmin(g[o:]))
    return seg[k]


X, Y = [], []
for lo, hi in bounds:
    p = block_xy(lo)
    if p is None:
        continue
    X.append(p)
    Y.append(grasp_pose(act[lo:hi]))
X, Y = np.array(X), np.array(Y)
n = len(X)
print("episodes usable: %d" % n)
print()


def design(P, quad):
    cols = [P[:, 0], P[:, 1], np.ones(len(P))]
    if quad:
        cols += [P[:, 0] ** 2, P[:, 1] ** 2, P[:, 0] * P[:, 1]]
    return np.column_stack(cols)


for quad, name in ((False, "linear   (x, y)"), (True, "quadratic (x, y, x2, y2, xy)")):
    err = np.zeros_like(Y)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        A = design(X[tr], quad)
        coef, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
        err[i] = design(X[i:i + 1], quad)[0] @ coef - Y[i]
    print("=== %s ===" % name)
    print("%-9s %10s %10s %10s" % ("joint", "spread sd", "LOO error", "ratio"))
    for j, nm in enumerate(J):
        sd = Y[:, j].std()
        e = np.abs(err[:, j]).mean()
        print("%-9s %10.2f %10.2f %10.2f" % (nm, sd, e, e / max(sd, 1e-9)))
    print()

print("ratio well below 1 means the pose is largely computable from the pixel")
print("position. Around 1 means the mapping carries no information.")
