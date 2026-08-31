"""Does the policy handle all three recorded block positions, or only one?

On the robot it reaches exactly one of the three positions and ignores the rest,
and it does respond to the block being there -- put it back and it grasps again.
So vision is being used for presence but not for location.

Ask the model directly: for episodes from each recorded group, feed the opening
frame and compare the reach it plans against the reach the operator actually
used in that episode. A group the policy handles well will show a small error;
groups it has collapsed away from will show a large one, biased toward whichever
position it does go to.

Groups are the recording order: 0-24, 25-49, 50-74.

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

CK, ROOT, LIFT, PAN = "policies/act_v3_100k", "datasets/so101_v3", 1, 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ds = LeRobotDataset("local/v2", root=ROOT)
edf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))
ddf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))]).sort_values(["episode_index", "frame_index"])
act = np.stack(ddf["action"].to_numpy())

policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})


def truth(lo, hi):
    """Pan angle at the deepest descent -- where the operator actually went."""
    seg = act[lo:hi]
    return seg[int(np.argmin(seg[:, LIFT])), PAN]


def planned(lo, horizon=150):
    """Roll the policy forward open loop from the opening frame, chunk after
    chunk, and report the furthest pan it commits to. One chunk is only 1.67 s,
    far short of a reach, so a single chunk would understate the target."""
    s = ds[lo]
    b = {k: s[k][None] for k in
         ("observation.state", "observation.images.top", "observation.images.wrist")}
    with torch.no_grad():
        policy.reset()
        o = pre(b)
        c = np.array([post(policy.select_action(o))[0].cpu().numpy()
                      for _ in range(min(horizon, cfg.chunk_size))])
    j = int(np.argmin(c[:, LIFT]))
    return c[j, PAN], c[:, PAN].max()


print("policy: %s" % CK)
print()
print("%-10s %10s %10s %10s %10s" % ("group", "true pan", "planned", "error", "n"))
rows = {}
for name, sl in (("ep 0-24", range(0, 25)), ("ep 25-49", range(25, 50)), ("ep 50-74", range(50, 75))):
    ts, ps = [], []
    for e in sl:
        lo, hi = bounds[e]
        ts.append(truth(lo, hi))
        ps.append(planned(lo)[0])
    ts, ps = np.array(ts), np.array(ps)
    rows[name] = (ts, ps)
    print("%-10s %10.2f %10.2f %10.2f %10d"
          % (name, ts.mean(), ps.mean(), np.abs(ts - ps).mean(), len(ts)))
print()
allt = np.concatenate([r[0] for r in rows.values()])
allp = np.concatenate([r[1] for r in rows.values()])
print("planned pan overall: mean %.2f  sd %.2f" % (allp.mean(), allp.std()))
print("true    pan overall: mean %.2f  sd %.2f" % (allt.mean(), allt.std()))
print("correlation planned vs true: %.3f" % float(np.corrcoef(allp, allt)[0, 1]))
print()
print("A planned sd far below the true sd means one target for every input.")
print("Which group it lands nearest tells you which position it collapsed onto.")
