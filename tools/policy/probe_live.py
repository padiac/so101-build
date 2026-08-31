"""Compare the live rollout frame against training, and ask the policy about it.

Established: on training frames this policy tracks the block position well
(planned vs true pan correlate 0.829; per-group errors 4.2-4.9 units). On the
robot it reaches only one place. So the gap is in what the live observation
looks like, not in what the model learned.

Two questions, and they point at different causes:
  1. Does the live overhead frame resemble the training frames, and is the block
     detected where it actually is? A shifted or re-exposed camera would move the
     apparent position and be read as a different location.
  2. Fed that live frame, where does the policy plan to reach -- the block's
     actual position, or the one place it keeps going to?

ASCII output only.
"""

import glob

import cv2
import numpy as np
import pandas as pd
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

CK, TRAIN, ROLL = "policies/act_v3_100k", "datasets/so101_v3", "datasets/rollout_probe"
LIFT, PAN, FPS = 1, 0, 30
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def block_xy(img_chw):
    a = (img_chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    hsv = cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, (18, 90, 90), (38, 255, 255)),
                         cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return None, 0
    x, y, w, h = cv2.boundingRect(max(c, key=cv2.contourArea))
    return (x + w / 2, y + h / 2), w * h


tr = LeRobotDataset("local/train", root=TRAIN)
ro = LeRobotDataset("local/roll", root=ROLL)
edf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{TRAIN}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))
ddf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{TRAIN}/data/**/*.parquet", recursive=True))]).sort_values(["episode_index", "frame_index"])
act = np.stack(ddf["action"].to_numpy())

# where the block sat in training, and where the operator reached
tp, tt = [], []
for lo, hi in bounds:
    p, _ = block_xy(tr[lo]["observation.images.top"])
    if p is None:
        continue
    seg = act[lo:hi]
    tp.append(p)
    tt.append(seg[int(np.argmin(seg[:, LIFT])), PAN])
tp, tt = np.array(tp), np.array(tt)

lp, larea = block_xy(ro[0]["observation.images.top"])
print("=== 1. where the block appears ===")
print("training blocks : x %.0f..%.0f   y %.0f..%.0f"
      % (tp[:, 0].min(), tp[:, 0].max(), tp[:, 1].min(), tp[:, 1].max()))
if lp is None:
    print("live frame      : BLOCK NOT DETECTED -- the policy has nothing to aim at")
else:
    print("live frame      : x %.0f   y %.0f   area %d px" % (lp[0], lp[1], larea))
    inside = (tp[:, 0].min() - 20 <= lp[0] <= tp[:, 0].max() + 20 and
              tp[:, 1].min() - 20 <= lp[1] <= tp[:, 1].max() + 20)
    print("inside the region covered by training: %s" % ("yes" if inside else "NO"))
    d = np.linalg.norm(tp - np.array(lp), axis=1)
    near = np.argsort(d)[:5]
    print("nearest training episodes: %s  (distances %s)"
          % (near.tolist(), np.round(d[near], 1).tolist()))
    print("their reach angles       : %s" % np.round(tt[near], 1).tolist())
print()

# brightness, to catch an exposure shift
lt = np.stack([tr[b[0]]["observation.images.top"].mean().item() for b in bounds])
print("overhead brightness: training mean %.3f   live %.3f"
      % (lt.mean(), float(ro[0]["observation.images.top"].mean())))
print()

print("=== 2. what the policy plans from that live frame ===")
policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})
s = ro[0]
b = {k: s[k][None] for k in ("observation.state", "observation.images.top", "observation.images.wrist")}
with torch.no_grad():
    policy.reset()
    o = pre(b)
    c = np.array([post(policy.select_action(o))[0].cpu().numpy() for _ in range(cfg.chunk_size)])
j = int(np.argmin(c[:, LIFT]))
print("planned pan at the deepest point of the chunk: %.2f" % c[j, PAN])
if lp is not None:
    A = np.column_stack([tp[:, 0], tp[:, 1], np.ones(len(tp))])
    coef, *_ = np.linalg.lstsq(A, tt, rcond=None)
    expect = float(np.array([lp[0], lp[1], 1.0]) @ coef)
    print("expected from the block's position:           %.2f" % expect)
    print("error:                                        %.2f" % abs(c[j, PAN] - expect))
print()
print("Block outside the training region -> nothing was ever demonstrated there.")
print("Block inside it but the plan is wrong -> the live image reads differently.")
print("Plan correct but the arm went elsewhere -> the fault is downstream of the policy.")
