"""Which half of the observation makes the policy go inert -- the images or the state?

Established so far:
  * fed a TRAINING observation, this policy plans a large reach (path ~490)
  * fed the ROLLOUT observation, it plans micro-jitter, and the robot's actual
    commands match that offline prediction (analyze_rollout Q2, MAE ~0.45)

So the difference lives in the observation. An observation has two parts and I
have only ever examined the images. This swaps them independently:

    train image + train state   -> known to move
    roll  image + roll  state   -> known to be inert
    train image + roll  state   -> ?
    roll  image + train state   -> ?

Whichever swap restores motion identifies the culprit. Total planned path is
the measure: sum of |per-step change| over the 100-step chunk.

ASCII output only.
"""

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

CK = "policies/act_trim_100k"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})

train = LeRobotDataset("local/train", root="datasets/so101_pickplace")
roll = LeRobotDataset("local/roll", root="datasets/rollout_probe")

t = train[0]
r = roll[0]

print("train state:", np.round(t["observation.state"].numpy(), 2))
print("roll  state:", np.round(r["observation.state"].numpy(), 2))
print("difference :", np.round((r["observation.state"] - t["observation.state"]).numpy(), 2))
print()


def path(img_src, state_src, h=100):
    b = {
        "observation.state": state_src["observation.state"][None],
        "observation.images.top": img_src["observation.images.top"][None],
        "observation.images.wrist": img_src["observation.images.wrist"][None],
    }
    with torch.no_grad():
        policy.reset()
        o = pre(b)
        c = np.array([post(policy.select_action(o))[0].cpu().numpy() for _ in range(h)])
    return np.abs(np.diff(c, axis=0)).sum(), c


print("%-34s %12s" % ("combination", "planned path"))
print("-" * 48)
for label, im, st in [
    ("train image + train state", t, t),
    ("roll  image + roll  state", r, r),
    ("train image + roll  state", t, r),
    ("roll  image + train state", r, t),
]:
    p, _ = path(im, st)
    print("%-34s %12.1f  %s" % (label, p, "MOVES" if p > 50 else "inert"))
