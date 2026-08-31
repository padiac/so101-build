"""Does where the block appears predict where the arm has to reach?

The policy ignores its images -- swapping them moves its plan by 0.85 while the
real targets span 16.23. Before blaming the model, check the data can be learned
at all: if the block's position in the overhead frame does not predict the grasp
angle in our own recordings, no network could recover that mapping, and the
fault is in how the data was captured.

Correlates the block's pixel position at the start of each episode against the
shoulder_pan the operator actually reached to.

ASCII output only.
"""

import glob

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT, FPS, PAN, GRIP = "datasets/so101_v3", 30, 0, 5

ds = LeRobotDataset("local/v2", root=ROOT)
edf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))
ddf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))]).sort_values(["episode_index", "frame_index"])
act = np.stack(ddf["action"].to_numpy())


def block_xy(idx):
    a = (ds[idx]["observation.images.top"].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    hsv = cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 90, 90), (38, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return None
    x, y, w, h = cv2.boundingRect(max(c, key=cv2.contourArea))
    return x + w / 2, y + h / 2


def grasp_frame(g):
    o = int(np.argmax(g))
    thr = g.min() + 0.30 * (g.max() - g.min())
    hold = int(0.7 * FPS)
    return next((i for i in range(o, len(g) - hold) if (g[i:i + hold] < thr).all()),
                int(o + np.argmin(g[o:])))


rows = []
for e, (lo, hi) in enumerate(bounds):
    p = block_xy(lo)
    if p is None:
        continue
    k = grasp_frame(act[lo:hi, GRIP])
    rows.append((e, p[0], p[1], act[lo + k, PAN], act[lo + k, 1], act[lo + k, 2]))

d = np.array([r[1:] for r in rows])
px, py, pan, lift, elbow = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]
print("episodes with a detected block: %d / %d" % (len(rows), len(bounds)))
print()
print("%-22s %8s %8s" % ("", "corr", "|corr|"))
for nm, a_, b_ in (("block px  vs shoulder_pan", px, pan),
                   ("block py  vs shoulder_pan", py, pan),
                   ("block px  vs shoulder_lift", px, lift),
                   ("block py  vs shoulder_lift", py, lift),
                   ("block px  vs elbow_flex", px, elbow),
                   ("block py  vs elbow_flex", py, elbow)):
    c = float(np.corrcoef(a_, b_)[0, 1])
    print("%-22s %8.3f %8.3f %s" % (nm, c, abs(c), "strong" if abs(c) > 0.7 else
                                    ("some" if abs(c) > 0.4 else "weak")))
print()
# best linear fit of pan from the pixel position
A = np.column_stack([px, py, np.ones(len(px))])
coef, *_ = np.linalg.lstsq(A, pan, rcond=None)
pred = A @ coef
resid = pan - pred
print("predicting shoulder_pan from block pixel position:")
print("  R^2               %.3f" % (1 - resid.var() / pan.var()))
print("  residual sd       %.2f  (pan sd is %.2f)" % (resid.std(), pan.std()))
print()
print("A high R^2 means the mapping is there to be learned and the model failed")
print("to pick it up. A low one means the recordings do not contain it.")
