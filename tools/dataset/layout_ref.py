"""Contact sheet of the scene layout at the start of every training episode.

The rollout put the black bowl far closer to the gripper than any demonstration
did, which is why the wrist view fell outside the training distribution. These
are the layouts the policy actually knows; matching one of them removes that
variable before anything else is changed.

Uses the overhead feed. Note the keys are swapped relative to physical reality:
observation.images.wrist is the overhead camera and observation.images.top is
the wrist-mounted one. The swap is consistent between training and deployment,
so it is harmless -- renaming would require retraining -- but it is confusing.

ASCII output only.
"""

import glob

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

OVERHEAD = "observation.images.wrist"     # physically the top camera

train = LeRobotDataset("local/train", root="datasets/so101_v3")
edf = pd.concat([pd.read_parquet(f) for f in
                 sorted(glob.glob("datasets/so101_v3/meta/episodes/**/*.parquet",
                                  recursive=True))]).sort_values("episode_index")
starts = edf["dataset_from_index"].astype(int).tolist()

tiles = []
for e, i in enumerate(starts):
    a = (train[int(i)][OVERHEAD].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    img = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (213, 160))
    cv2.putText(img, "ep%d" % e, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, "ep%d" % e, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(img)

cols = 6
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
cv2.imwrite("layout_ref.png", np.vstack(rows))
print("wrote layout_ref.png  (%d episode starts, overhead view)" % len(starts))
