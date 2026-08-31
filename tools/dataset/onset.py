"""When does motion actually start in each recorded episode?

WHY
    Closed-loop eval from the training start pose does nothing at all, while
    the same policy plans a large reach from a slightly different pose. That
    points at dead time at the head of every episode: recording begins, the
    operator gets ready, and only then starts moving.

    A policy trained on that sees the idle pose paired with "stay still" in
    some frames and "start reaching" in others. It cannot tell which from the
    image alone, so it averages -- and averaging a wait with a reach gives
    almost no motion. Replanning every 167 ms re-makes that same decision
    forever, so it never escapes.

    With n_action_steps=100 the arm instead commits to a whole chunk and does
    move -- which is exactly the difference between the first eval run and the
    second.

ASCII output only.
"""

import glob

import numpy as np
import pandas as pd

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ROOT = "datasets/so101_v3"
THRESH = 3.0        # units of cumulative motion that count as "started"

files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files]).sort_values(["episode_index", "frame_index"])
act = np.stack(df["action"].to_numpy())
ep = df["episode_index"].to_numpy()

onsets, lengths = [], []
for e in np.unique(ep):
    a = act[ep == e]
    d = np.abs(a - a[0]).max(axis=1)      # furthest any joint has moved from frame 0
    idx = np.argmax(d > THRESH) if (d > THRESH).any() else len(a)
    onsets.append(idx)
    lengths.append(len(a))

onsets = np.array(onsets)
lengths = np.array(lengths)

print("episodes %d" % len(onsets))
print()
print("frames before motion exceeds %.0f units:" % THRESH)
print("  min %d   median %d   mean %.0f   max %d"
      % (onsets.min(), int(np.median(onsets)), onsets.mean(), onsets.max()))
print("  in seconds at 30 fps: min %.1f  median %.1f  max %.1f"
      % (onsets.min()/30, np.median(onsets)/30, onsets.max()/30))
print()
print("dead frames total  %d / %d  = %.1f%% of the dataset"
      % (onsets.sum(), lengths.sum(), 100*onsets.sum()/lengths.sum()))
print()
print("per-episode (frames of dead time / total):")
for i, (o, L) in enumerate(zip(onsets, lengths)):
    bar = "#" * int(o / 5)
    print("  ep %2d  %4d / %4d  %s" % (i, o, L, bar))
