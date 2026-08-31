"""Does the policy actually look at the camera, or replay one trajectory?

Observed: the arm grasps reliably at one of the three recorded block positions
and drives to that same spot regardless of where the block actually is.

That is what a policy ignoring its images looks like. Every episode starts from
the same home pose, so proprioception alone predicts the average trajectory --
and regressing on the state is far easier than reading the block's position out
of the pixels. The images become decoration.

Test: hold the state fixed and swap in the opening frames from episodes at
different block positions. If the planned trajectory barely moves, vision is not
being used. shoulder_pan is the informative joint -- it is what aims the arm
left or right.

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

CK, ROOT, FPS = "policies/act_v3_100k", "datasets/so101_v3", 30
PAN, GRIP = 0, 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ds = LeRobotDataset("local/v2", root=ROOT)
edf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))
ddf = pd.concat([pd.read_parquet(f) for f in sorted(
    glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))]).sort_values(["episode_index", "frame_index"])
act = np.stack(ddf["action"].to_numpy())


def grasp_idx(g):
    o = int(np.argmax(g))
    thr = g.min() + 0.30 * (g.max() - g.min())
    hold = int(0.7 * FPS)
    return next((i for i in range(o, len(g) - hold) if (g[i:i + hold] < thr).all()),
                int(o + np.argmin(g[o:])))


# where each episode actually reached, as recorded
reach = []
for lo, hi in bounds:
    k = grasp_idx(act[lo:hi, GRIP])
    reach.append(act[lo + k, PAN])
reach = np.array(reach)

print("recorded shoulder_pan at the grasp, by episode block")
for name, sl in (("ep  0-24", slice(0, 25)), ("ep 25-49", slice(25, 50)), ("ep 50-74", slice(50, 75))):
    r = reach[sl]
    print("  %-9s mean %7.2f   sd %5.2f   range %7.2f .. %7.2f"
          % (name, r.mean(), r.std(), r.min(), r.max()))
print()

policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})


def plan(state_src, img_src, h=None):
    h = h or cfg.chunk_size
    b = {"observation.state": ds[state_src]["observation.state"][None]}
    for cam in ("top", "wrist"):
        b["observation.images.%s" % cam] = ds[img_src]["observation.images.%s" % cam][None]
    with torch.no_grad():
        policy.reset()
        o = pre(b)
        return np.array([post(policy.select_action(o))[0].cpu().numpy() for _ in range(h)])


# one representative episode start from each block
reps = [bounds[10][0], bounds[35][0], bounds[60][0]]
labels = ["blockA(ep10)", "blockB(ep35)", "blockC(ep60)"]

print("planned shoulder_pan at the end of the chunk")
print("rows = state from, columns = images from")
print("%-16s %14s %14s %14s" % ("", labels[0], labels[1], labels[2]))
grid = np.zeros((3, 3))
for i, s in enumerate(reps):
    row = []
    for j, im in enumerate(reps):
        c = plan(s, im)
        grid[i, j] = c[-1, PAN]
        row.append("%14.2f" % c[-1, PAN])
    print("%-16s %s" % (labels[i], "".join(row)))
print()

across_images = grid.std(axis=1).mean()     # varying the image, state fixed
across_states = grid.std(axis=0).mean()     # varying the state, image fixed
print("spread when only the IMAGE changes : %.2f" % across_images)
print("spread when only the STATE changes : %.2f" % across_states)
print("spread of the recorded targets     : %.2f" % reach.std())
print()
print("If changing the image moves the plan far less than the recorded targets")
print("vary, the policy is not localising from vision.")
