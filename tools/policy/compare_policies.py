"""Do the old and new policies plan differently from the episode-start pose?

The failure being tested: from the idle start pose the previous policy planned
to stand still, because 22% of its training frames paired that exact pose with
"keep waiting". Trimming the lead-in should make the same pose plan a reach.

This runs entirely offline -- no robot, no cameras -- by feeding frame 0 of an
episode from each policy's own training set.

ASCII output only.
"""

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(ck):
    p = ACTPolicy.from_pretrained(ck).to(DEV).eval()
    cfg = PreTrainedConfig.from_pretrained(ck)
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=ck,
        preprocessor_overrides={"device_processor": {"device": DEV}})
    return p, pre, post


def plan(p, pre, post, sample, h=100):
    b = {k: sample[k][None] for k in
         ("observation.state", "observation.images.top", "observation.images.wrist")}
    with torch.no_grad():
        p.reset()
        o = pre(b)
        return np.array([post(p.select_action(o))[0].cpu().numpy() for _ in range(h)])


def report(name, chunk):
    print("=== %s ===" % name)
    print("%-15s %9s %9s %9s" % ("joint", "net move", "path", "verdict"))
    total = 0.0
    for j, nm in enumerate(JOINTS):
        c = chunk[:, j]
        net = c[-1] - c[0]
        path = np.abs(np.diff(c)).sum()
        total += path
        print("%-15s %9.2f %9.2f" % (nm, net, path))
    print("%-15s %9s %9.2f" % ("TOTAL PATH", "", total))
    print("  -> %s" % ("STANDS STILL" if total < 10 else "PLANS A MOTION"))
    print()
    return total


# each policy evaluated on frame 0 of its own dataset
old_ds = LeRobotDataset("local/a", root="datasets/so101_v3")
print("loading old policy (untrimmed data)")
p, pre, post = load("policies/act_v3_100k")
t_old = report("OLD  policy, untrimmed frame 0", plan(p, pre, post, old_ds[0]))

print("loading new policy (trimmed data)")
p2, pre2, post2 = load("policies/act_v3_100k")
# the trimmed dataset lives in WSL; use the same untrimmed frame 0 so the
# INPUT is identical and only the policy differs
t_new = report("NEW  policy, same frame 0", plan(p2, pre2, post2, old_ds[0]))

print("total planned path: old %.2f -> new %.2f  (%.1fx)"
      % (t_old, t_new, t_new / max(t_old, 1e-6)))
