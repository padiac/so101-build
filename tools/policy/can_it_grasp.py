"""Does the policy know how to descend and close, when shown the moment before?

Observed on the robot: it lifts, roughly lines up over the block, stops, backs
off, and starts the approach again -- it never descends and never closes. The
question is whether that behaviour is absent from the model or merely never
reached.

So put the model in the state where the demonstration is about to descend and
grasp, using real frames from the training set, and read out what it plans. If
the chunk descends and closes the gripper, the behaviour was learned and the
failure is that the robot never arrives in that state distribution. If it does
not, the behaviour was never learned.

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
ROOT, FPS = "datasets/so101_v3", 30
GRIP, LIFT = 5, 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"

policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})

ds = LeRobotDataset("local/train", root=ROOT)
edf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))
ddf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))]).sort_values(["episode_index", "frame_index"])
act = np.stack(ddf["action"].to_numpy())


def grasp_frame(g):
    o = int(np.argmax(g))
    thr = g.min() + 0.30 * (g.max() - g.min())
    hold = int(0.7 * FPS)
    for i in range(o, len(g) - hold):
        if (g[i:i + hold] < thr).all():
            return i
    return int(o + np.argmin(g[o:]))


def plan(i):
    s = ds[int(i)]
    b = {k: s[k][None] for k in
         ("observation.state", "observation.images.top", "observation.images.wrist")}
    with torch.no_grad():
        policy.reset()
        o = pre(b)
        return np.array([post(policy.select_action(o))[0].cpu().numpy() for _ in range(100)])


# episodes where the grasp frame was clean (block visible between the fingers)
GOOD = [1, 2, 3, 5, 6, 7]
print("policy: %s" % CK)
print()
print("%4s %10s %26s %26s" % ("ep", "offset", "gripper over chunk", "shoulder_lift over chunk"))
print("%4s %10s %8s %8s %8s %8s %8s %8s" % ("", "", "start", "min", "closes?", "start", "min", "descends?"))
for e in GOOD:
    lo, hi = bounds[e]
    k = grasp_frame(act[lo:hi, GRIP])
    for off_s in (-1.5, -0.5, 0.0):
        i = lo + max(0, k + int(off_s * FPS))
        c = plan(i)
        g, l = c[:, GRIP], c[:, LIFT]
        closes = "YES" if (g.min() < g[0] - 5) else "no"
        # in this dataset lower shoulder_lift means the arm is further down
        desc = "YES" if (l.min() < l[0] - 5) else "no"
        print("%4d %10.1f %8.1f %8.1f %8s %8.1f %8.1f %8s"
              % (e, off_s, g[0], g.min(), closes, l[0], l.min(), desc))
    print()
