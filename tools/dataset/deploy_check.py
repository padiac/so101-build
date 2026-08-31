"""Measure two deployment-side suspects from the training data itself.

1) max_relative_target clamping
   robot.ps1 passes --robot.max_relative_target=15, which caps each joint's
   per-step command delta. If the policy needs bigger steps than that, the
   commanded trajectory gets distorted and the arm moves oddly.

2) start-pose distribution
   Every training episode started from wherever the leader happened to be.
   At eval the arm starts wherever it was left. If that is outside the
   training start distribution, the policy is out of distribution from
   step 0 and never recovers.

ASCII output only -- the Windows console is GBK and this gets piped around.
"""

import numpy as np
import pandas as pd
import glob

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ROOT = "datasets/so101_v3"

files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files]).sort_values(["episode_index", "frame_index"])

act = np.stack(df["action"].to_numpy())
state = np.stack(df["observation.state"].to_numpy())
ep = df["episode_index"].to_numpy()

print("frames %d   episodes %d" % (len(df), len(np.unique(ep))))
print()

# ---- 1) per-step deltas, computed within each episode ----
d = []
for e in np.unique(ep):
    m = ep == e
    d.append(np.abs(np.diff(act[m], axis=0)))
d = np.concatenate(d)

print("=== per-step |delta action| vs max_relative_target=15 ===")
print("%-15s %8s %8s %8s %8s" % ("joint", "median", "p95", "p99", "max"))
worst = 0.0
for j, nm in enumerate(JOINTS):
    q = np.percentile(d[:, j], [50, 95, 99]), d[:, j].max()
    worst = max(worst, q[1])
    print("%-15s %8.2f %8.2f %8.2f %8.2f" % (nm, q[0][0], q[0][1], q[0][2], q[1]))
over = (d > 15).mean() * 100
print()
print("fraction of all per-step deltas above 15 : %.2f%%" % over)
print("largest single-step delta anywhere       : %.2f" % worst)

# ---- 2) start pose distribution ----
print()
print("=== episode start pose (observation.state at frame 0) ===")
starts = np.stack([state[ep == e][0] for e in np.unique(ep)])
print("%-15s %9s %9s %9s %9s" % ("joint", "min", "mean", "max", "spread"))
for j, nm in enumerate(JOINTS):
    c = starts[:, j]
    print("%-15s %9.2f %9.2f %9.2f %9.2f" % (nm, c.min(), c.mean(), c.max(), c.max() - c.min()))
print()
print("Park the arm inside these ranges before running eval.")
