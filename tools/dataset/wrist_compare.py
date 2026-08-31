"""Side-by-side: what the wrist camera saw in training vs during the rollout.

The wrist camera is bolted to the follower, so the same arm pose must produce
the same view. Q1 of analyze_rollout.py put the rollout's wrist frames out of
the training distribution (0.820 against a 0.922 train-train baseline) while
the top camera was fine, which can only mean the camera moved relative to the
arm -- shifted, rotated, or refocused -- since the data was recorded.

Both frames below are taken with the arm at the same start pose, so anything
that differs is the camera, not the robot.

ASCII output only.
"""

import glob

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_bgr(t):
    a = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


train = LeRobotDataset("local/train", root="datasets/so101_v3")
roll = LeRobotDataset("local/roll", root="datasets/rollout_probe")

# episode starts in training = arm at the home pose, same as the rollout start
edf = pd.concat([pd.read_parquet(f) for f in
                 sorted(glob.glob("datasets/so101_v3/meta/episodes/**/*.parquet",
                                  recursive=True))]).sort_values("episode_index")
starts = edf["dataset_from_index"].astype(int).tolist()

panels, labels = [], []
for i in starts[:3]:
    panels.append(to_bgr(train[int(i)]["observation.images.wrist"]))
    labels.append("TRAIN ep-start")
for i in (0, len(roll) // 2, len(roll) - 1):
    panels.append(to_bgr(roll[int(i)]["observation.images.wrist"]))
    labels.append("ROLLOUT f%d" % i)

for p, l in zip(panels, labels):
    cv2.putText(p, l, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(p, l, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

top = np.hstack(panels[:3])
bot = np.hstack(panels[3:])
img = np.vstack([top, bot])
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
cv2.imwrite("wrist_compare.png", img)
print("wrote wrist_compare.png  (top row = training, bottom row = rollout)")

# same for the top camera, as a control that did pass Q1
panels = [to_bgr(train[int(i)]["observation.images.top"]) for i in starts[:3]] + \
         [to_bgr(roll[int(i)]["observation.images.top"]) for i in (0, len(roll)//2, len(roll)-1)]
img2 = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2))
cv2.imwrite("top_compare.png", img2)
print("wrote top_compare.png    (control -- this camera passed Q1)")
