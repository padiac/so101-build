"""How much of the planned motion happens in the first few steps of a chunk?

The policy plans a 100-step trajectory whose total path is ~580 units. On the
robot with n_action_steps=5 only the first 5 of those steps are ever executed;
then the chunk is discarded and a fresh one is planned. If the trajectory
barely moves in its opening steps, the arm never leaves the start pose no
matter how many times it replans -- which matches the observed micro-jitter,
and matches the first run (n_action_steps=100) moving while this one does not.

This prints cumulative displacement as a function of position in the chunk.

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

policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})

roll = LeRobotDataset("local/roll", root="datasets/rollout_probe")

# average over several rollout observations so this is not one lucky frame
idx = np.linspace(0, len(roll) - 1, 8).astype(int)
chunks = []
with torch.no_grad():
    for i in idx:
        s = roll[int(i)]
        b = {k: s[k][None] for k in
             ("observation.state", "observation.images.top", "observation.images.wrist")}
        policy.reset()
        o = pre(b)
        chunks.append(np.array([post(policy.select_action(o))[0].cpu().numpy()
                                for _ in range(100)]))
chunks = np.array(chunks)                       # (N, 100, 6)

start = chunks[:, :1, :]                        # first action of each chunk
disp = np.abs(chunks - start).max(axis=2)       # (N, 100) furthest any joint has moved

print("displacement from the chunk's own first action, averaged over %d chunks" % len(chunks))
print("%8s %14s %14s" % ("step", "mean max-joint", "as %% of final"))
final = disp[:, -1].mean()
for k in (1, 2, 4, 9, 14, 19, 29, 49, 74, 99):
    v = disp[:, k].mean()
    print("%8d %14.2f %13.1f%%" % (k + 1, v, 100 * v / max(final, 1e-6)))
print()
print("executed slice at n_action_steps=5 -> only the first 5 rows matter")
print()

# per-joint view of the first 5 steps
print("first 5 steps, per joint (mean absolute change from step 1):")
print("%-10s %8s %8s %8s %8s" % ("joint", "step2", "step3", "step4", "step5"))
for j, nm in enumerate(JOINTS):
    d = np.abs(chunks[:, :5, j] - chunks[:, :1, j]).mean(axis=0)
    print("%-10s %8.2f %8.2f %8.2f %8.2f" % (nm, d[1], d[2], d[3], d[4]))
