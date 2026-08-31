"""How fast does the predicted action chunk drift from the recorded one?

WHY
    ACT predicts chunk_size=100 actions at once. With n_action_steps=100 and no
    temporal ensembling, lerobot executes all 100 open loop -- 3.3 s at 30 fps
    with no new camera input. Any error compounds over that window.

    Step-1 accuracy (which eval_offline.py measures) says nothing about this.
    A policy can be excellent at "what do I do right now" and still fail on the
    robot because step 80 of its plan is nonsense.

    So: compare the whole predicted chunk against the next 100 recorded actions
    and watch the error as a function of position in the chunk.

ASCII output only.
"""

import glob
import sys

import numpy as np
import pandas as pd
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

CKPT = "policies/act_pickplace_100k"
ROOT = "datasets/so101_pickplace"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

ds = LeRobotDataset("local/so101_pickplace", root=ROOT)
policy = ACTPolicy.from_pretrained(CKPT)
policy.to("cpu"); policy.eval()
cfg = PreTrainedConfig.from_pretrained(CKPT)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": "cpu"}},
)

# recorded actions laid out per episode so we can grab 100 consecutive steps
files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files]).sort_values(["episode_index", "frame_index"])
act = np.stack(df["action"].to_numpy())
ep = df["episode_index"].to_numpy()
gidx = np.arange(len(df))

H = 100
rng = np.random.default_rng(0)
# pick starts that have 100 more frames inside the same episode
cands = [i for i in gidx[:-H] if ep[i] == ep[i + H - 1]]
picks = rng.choice(cands, size=min(N, len(cands)), replace=False)

errs = []
with torch.no_grad():
    for i in picks:
        s = ds[int(i)]
        batch = {
            "observation.state": s["observation.state"][None],
            "observation.images.top": s["observation.images.top"][None],
            "observation.images.wrist": s["observation.images.wrist"][None],
        }
        policy.reset()
        obs = pre(batch)
        chunk = []
        for _ in range(H):                 # drains the internal queue = the whole chunk
            chunk.append(post(policy.select_action(obs))[0].numpy())
        chunk = np.array(chunk)            # (H, 6)
        gt = act[i:i + H]                  # (H, 6)
        errs.append(np.abs(chunk - gt).mean(axis=1))

errs = np.array(errs)                      # (N, H)
print("chunks %d   horizon %d   (%.1f s of open-loop execution at 30 fps)"
      % (len(errs), H, H / 30))
print()
print("%8s %10s %10s" % ("step", "mean MAE", "p90 MAE"))
for k in (0, 4, 9, 19, 29, 49, 74, 99):
    print("%8d %10.2f %10.2f" % (k + 1, errs[:, k].mean(), np.percentile(errs[:, k], 90)))
print()
print("For scale, joint travel ranges are 45-190 units.")
print("If the error at step 100 is a large fraction of that, running the whole")
print("chunk open loop is the problem, not the model.")
