"""How near did the rollout get to a demonstrated pre-grasp pose?

The policy plans a descent and a gripper close whenever it is shown a real
pre-grasp frame, so the behaviour exists. On the robot it lines up roughly,
stops, and retreats -- meaning it never enters that region of state space.

This measures the gap directly: for every rollout frame, the distance in joint
space to the nearest demonstrated pre-grasp pose, against the spread of those
demonstrated poses themselves. A gap comparable to that spread means more
demonstrations around the same positions should close it; a gap far larger
means the approach is ending somewhere else entirely.

ASCII output only.
"""

import glob

import numpy as np
import pandas as pd

JOINTS = ["sh_pan", "sh_lift", "elbow", "wr_flex", "wr_roll", "grip"]
ROOT, ROLL, FPS, GRIP = "datasets/so101_v3", "datasets/rollout_probe", 30, 5


def load(root):
    df = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"{root}/data/**/*.parquet", recursive=True))])
    return df.sort_values(["episode_index", "frame_index"])


tdf = load(ROOT)
tact = np.stack(tdf["action"].to_numpy())
tst = np.stack(tdf["observation.state"].to_numpy())
tep = tdf["episode_index"].to_numpy()

pre = []
for e in np.unique(tep):
    m = tep == e
    g = tact[m, GRIP]
    o = int(np.argmax(g))
    thr = g.min() + 0.30 * (g.max() - g.min())
    hold = int(0.7 * FPS)
    k = next((i for i in range(o, len(g) - hold) if (g[i:i + hold] < thr).all()),
             int(o + np.argmin(g[o:])))
    k = max(0, k - int(0.5 * FPS))          # half a second before closing
    pre.append(tst[m][k])
pre = np.array(pre)

rst = np.stack(load(ROLL)["observation.state"].to_numpy())

# spread of the demonstrated pre-grasp poses: nearest neighbour among themselves
D = np.linalg.norm(pre[:, None, :] - pre[None, :, :], axis=2)
np.fill_diagonal(D, np.inf)
spread = D.min(axis=1)

d = np.linalg.norm(rst[:, None, :] - pre[None, :, :], axis=2)
nn = d.min(axis=1)

print("demonstrated pre-grasp poses: %d" % len(pre))
print("  spacing between them   : median %.1f   max %.1f" % (np.median(spread), spread.max()))
print()
print("rollout frames: %d" % len(rst))
print("  distance to nearest    : min %.1f   median %.1f" % (nn.min(), np.median(nn)))
print()
best = int(np.argmin(nn))
print("closest approach at t=%.1fs, distance %.1f" % (best / FPS, nn.min()))
j = int(np.argmin(d[best]))
print("%-9s %10s %10s %9s" % ("joint", "rollout", "demo", "diff"))
for i, nm in enumerate(JOINTS):
    print("%-9s %10.2f %10.2f %9.2f" % (nm, rst[best][i], pre[j][i], rst[best][i] - pre[j][i]))
print()
print("If the closest approach is within the spacing of the demonstrations, the")
print("gap is one more demonstration wide. If it is several times larger, the")
print("approach is terminating somewhere the demonstrations never visited.")
