"""At the moment the arm stopped, did the policy still want to do anything?

The arm now reaches the block but comes up slightly misaligned, then holds
still until the clock runs out. Two very different causes:

  (a) it still plans a motion, but something downstream suppresses it;
  (b) it plans almost nothing -- the pose it reached is close to no
      demonstration, so there is no confident action to imitate and the
      averaged prediction collapses towards "stay".

(b) is the behaviour-cloning failure that more data fixes, and it tells you
exactly WHICH data: states near the final approach, slightly off-target.

This walks through the recorded rollout and reports how much motion the policy
plans from each point in time, next to how much the arm actually moved.

ASCII output only.
"""

import glob

import numpy as np
import pandas as pd
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

CK = "policies/act_v3_100k"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})

roll = LeRobotDataset("local/roll", root="datasets/rollout_probe")
df = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob("datasets/rollout_probe/data/**/*.parquet", recursive=True))])
df = df.sort_values("frame_index")
st = np.stack(df["observation.state"].to_numpy())

n = len(roll)
idx = np.linspace(0, n - 1, 20).astype(int)
print("rollout %d frames (%.1f s)" % (n, n / 30))
print()
print("%8s %8s %14s %16s" % ("frame", "t (s)", "planned path", "actual moved/0.5s"))
with torch.no_grad():
    for i in idx:
        s = roll[int(i)]
        b = {k: s[k][None] for k in
             ("observation.state", "observation.images.top", "observation.images.wrist")}
        policy.reset()
        o = pre(b)
        c = np.array([post(policy.select_action(o))[0].cpu().numpy() for _ in range(100)])
        planned = float(np.abs(np.diff(c, axis=0)).sum())
        lo, hi = max(0, i - 7), min(n - 1, i + 8)
        actual = float(np.abs(st[hi] - st[lo]).max())
        print("%8d %8.1f %14.1f %16.2f" % (i, i / 30, planned, actual))
print()
print("planned path collapsing towards zero while the clock runs means the pose")
print("it reached is unlike anything demonstrated -- a data gap, not a setting.")
