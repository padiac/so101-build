"""Do successive replans agree on where to go?

The arm reverses direction about every 5 frames, which is exactly the replan
interval, while each individual chunk plans a large coherent motion. Those two
facts are only compatible if consecutive replans disagree about the direction:
plan A moves the elbow one way, 167 ms later plan B moves it back, and the net
displacement is zero.

This takes observations 5 frames apart from the recorded rollout -- the same
spacing the robot actually replanned at -- and compares the signed direction of
each resulting chunk.

ASCII output only.
"""

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

CK = "policies/act_v3_100k"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
JOINTS = ["sh_pan", "sh_lift", "elbow", "wr_flex", "wr_roll", "grip"]
H = 20            # look at the near-term part of each plan

policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})

roll = LeRobotDataset("local/roll", root="datasets/rollout_probe")
idx = list(range(0, min(len(roll), 200), 5))       # same spacing as n_action_steps=5

dirs = []
with torch.no_grad():
    for i in idx:
        s = roll[int(i)]
        b = {k: s[k][None] for k in
             ("observation.state", "observation.images.top", "observation.images.wrist")}
        policy.reset()
        o = pre(b)
        c = np.array([post(policy.select_action(o))[0].cpu().numpy() for _ in range(H)])
        dirs.append(c[-1] - c[0])                 # signed near-term intent
dirs = np.array(dirs)                             # (N, 6)

print("replans analysed: %d, spaced %d frames apart (the real replan interval)" % (len(dirs), 5))
print()
print("%-10s %10s %10s %10s %12s" % ("joint", "mean", "std", "|mean|/std", "sign flips"))
for j, nm in enumerate(JOINTS):
    v = dirs[:, j]
    flips = int(np.sum(np.sign(v[1:]) != np.sign(v[:-1])))
    ratio = abs(v.mean()) / v.std() if v.std() > 1e-9 else float("inf")
    print("%-10s %10.2f %10.2f %10.2f %10d/%d" % (nm, v.mean(), v.std(), ratio, flips, len(v) - 1))
print()
print("|mean|/std well below 1, with many sign flips, means successive plans")
print("point in different directions and cancel out.")
print()
print("first 12 replans, elbow intent over the next %d steps:" % H)
print("  " + "  ".join("%+.0f" % v for v in dirs[:12, 2]))
