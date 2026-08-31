"""Is the model's error a pull toward the average, or is it random?

The arm handles the middle and far block positions but never the near one. Two
explanations need different responses:

  * errors that push low targets up and high targets down are regression toward
    the mean -- the model has not learned enough from the images and falls back
    on the average answer. That is underfitting, and it is fixed by training
    differently, not by collecting more data.

  * errors with no such structure would point at the recordings themselves.

Each episode is queried one chunk (1.67 s) before its grasp, so the plan's end
lands exactly on the moment being compared.

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

CK, ROOT, FPS, PAN, GRIP = "policies/act_v3_100k", "datasets/so101_v3", 30, 0, 5
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
H = cfg.chunk_size


def grasp_k(seg):
    g = seg[:, GRIP]
    o = int(np.argmax(g))
    thr = g.min() + 0.30 * (g.max() - g.min())
    hold = int(0.7 * FPS)
    k = next((i for i in range(o, len(g) - hold) if (g[i:i + hold] < thr).all()), None)
    return int(o + np.argmin(g[o:])) if k is None else k


true, pred, grp = [], [], []
for e, (lo, hi) in enumerate(bounds):
    seg = act[lo:hi]
    k = grasp_k(seg)
    i = lo + max(0, k - H)                 # one chunk before the grasp
    s = ds[i]
    b = {x: s[x][None] for x in
         ("observation.state", "observation.images.top", "observation.images.wrist")}
    with torch.no_grad():
        policy.reset()
        o = pre(b)
        c = np.array([post(policy.select_action(o))[0].cpu().numpy() for _ in range(H)])
    true.append(seg[k, PAN])
    pred.append(c[-1, PAN])
    grp.append(0 if e < 25 else (1 if e < 50 else 2))

true, pred, grp = np.array(true), np.array(pred), np.array(grp)
err = pred - true
mean_t = true.mean()

print("true grasp pan : mean %6.2f  sd %5.2f" % (mean_t, true.std()))
print("planned        : mean %6.2f  sd %5.2f" % (pred.mean(), pred.std()))
print("shrinkage (planned sd / true sd): %.2f   (1.0 = no pull to the average)"
      % (pred.std() / true.std()))
print()
print("%-24s %8s %8s %8s" % ("bucket by true value", "n", "mean err", "direction"))
order = np.argsort(true)
for name, sel in (("lowest third", order[:25]), ("middle third", order[25:50]),
                  ("highest third", order[50:])):
    e_ = err[sel]
    d = "pushed HIGHER" if e_.mean() > 2 else ("pushed LOWER" if e_.mean() < -2 else "-")
    print("%-24s %8d %8.2f %8s" % (name, len(sel), e_.mean(), d))
print()
print("%-14s %8s %8s %8s" % ("recorded group", "true", "planned", "err"))
for g, nm in ((0, "pos 1 (near)"), (1, "pos 2 (mid)"), (2, "pos 3 (far)")):
    m = grp == g
    print("%-14s %8.2f %8.2f %8.2f" % (nm, true[m].mean(), pred[m].mean(), err[m].mean()))
print()
slope = np.polyfit(true, pred, 1)[0]
print("slope of planned against true: %.2f   (1.0 = follows perfectly, 0 = ignores)" % slope)
