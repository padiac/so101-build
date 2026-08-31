"""What does the wrist camera see at the moment of the grasp?

A wrist camera is not there to localise the object -- the overhead view does
that. It earns its place in the last few centimetres: fine alignment and the
timing of the gripper closing. Judging it from the home pose, where it can only
ever see bare table, says nothing.

So sample it where it matters. The grasp is found as the frame where the
gripper command closes fastest, and the wrist view at that instant is what the
policy actually has to work with when it decides to close.

Note the dataset keys are the ones recorded in v1, where the names were still
swapped: observation.images.top holds the WRIST feed there.

ASCII output only.
"""

import glob

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

WRIST_KEY = "observation.images.top"     # v1 naming: this key holds the wrist feed
ROOT = "datasets/so101_v3"
GRIP = 5                                  # gripper is the last joint

train = LeRobotDataset("local/train", root=ROOT)
edf = pd.concat([pd.read_parquet(f) for f in
                 sorted(glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))])
edf = edf.sort_values("episode_index")
bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))

ddf = pd.concat([pd.read_parquet(f) for f in
                 sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))])
ddf = ddf.sort_values(["episode_index", "frame_index"])
act = np.stack(ddf["action"].to_numpy())

tiles = []
print("%4s %10s %10s" % ("ep", "grasp @s", "grip drop"))
for e, (lo, hi) in enumerate(bounds[:12]):
    g = act[lo:hi, GRIP]
    d = np.diff(g)
    k = int(np.argmin(d))                 # fastest closing
    print("%4d %10.1f %10.2f" % (e, k / 30, -d[k]))
    a = (train[lo + k][WRIST_KEY].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    img = cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), (320, 240))
    cv2.putText(img, "ep%d @%.1fs" % (e, k / 30), (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, "ep%d @%.1fs" % (e, k / 30), (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    tiles.append(img)

rows = [np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]
cv2.imwrite("grasp_view.png", np.vstack(rows))
print()
print("wrote grasp_view.png -- wrist view at the instant the gripper closes")
