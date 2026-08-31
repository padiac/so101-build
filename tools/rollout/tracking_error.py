"""Did the arm follow the commands it was given?

Successive replans agree completely on direction (0 sign flips over 40 replans,
elbow intent -59 +/- 4 every time), yet the arm stayed put. That rules out the
policy: it is issuing a firm, repeated instruction. The remaining question is
whether the joints executed it.

This compares the commanded action against the measured state in the recorded
rollout, and does the same for a training episode as a control -- during
teleoperation the follower tracked the leader well, so the training numbers
show what healthy tracking looks like on this hardware.

ASCII output only.
"""

import glob
import sys

import numpy as np
import pandas as pd

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load(root):
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))])
    df = df.sort_values(["episode_index", "frame_index"])
    return (np.stack(df["action"].to_numpy()),
            np.stack(df["observation.state"].to_numpy()),
            df["episode_index"].to_numpy())


def report(label, act, st):
    print("=== %s ===" % label)
    print("%-15s %10s %10s %10s %10s" % ("joint", "cmd range", "meas range", "mean err", "max err"))
    for j, nm in enumerate(JOINTS):
        a, s = act[:, j], st[:, j]
        err = np.abs(a - s)
        print("%-15s %10.2f %10.2f %10.2f %10.2f"
              % (nm, a.max() - a.min(), s.max() - s.min(), err.mean(), err.max()))
    print()


ra, rs, _ = load("datasets/rollout_probe")
report("ROLLOUT (policy driving)", ra, rs)

ta, ts, tep = load("datasets/so101_v3")
m = tep == 0
report("TRAINING episode 0 (teleop driving) -- control", ta[m], ts[m])

print("A command range far larger than the measured range means the joint was")
print("told to move and did not. Compare against the teleop control, where the")
print("follower did track.")
