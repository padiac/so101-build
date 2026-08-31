"""Did our code send what the model asked for, or did something change it?

The suspicion is a plumbing bug -- a default we pass differently from what
lerobot expects, or one of my patches altering the action on its way out. That
is testable without touching the robot: the rollout recorded every observation
the policy saw and every action that was sent, so replaying those observations
offline reproduces exactly what the model wanted.

An earlier version of this check compared only the first step of each chunk,
which is nearly the current position and therefore agrees almost by definition.
This one replays the whole run at the same replanning rhythm the robot used.

Matching means the fault is in what the model learned. Diverging means we alter
its output, and the per-joint pattern says where.

ASCII output only.
"""

import argparse
import glob

import numpy as np
import pandas as pd
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

J = ["sh_pan", "sh_lift", "elbow", "wr_flex", "wr_roll", "grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", default="datasets/rollout_probe")
    ap.add_argument("--ckpt", default="policies/act_v3_100k")
    ap.add_argument("--n-action-steps", type=int, default=100)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LeRobotDataset("local/roll", root=a.roll)
    df = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"{a.roll}/data/**/*.parquet", recursive=True))]).sort_values(
        ["episode_index", "frame_index"])
    sent = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    n = len(df)
    print("rollout %d frames (%.1f s), replanning every %d" % (n, n / 30, a.n_action_steps))

    policy = ACTPolicy.from_pretrained(a.ckpt).to(dev).eval()
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=a.ckpt,
        preprocessor_overrides={"device_processor": {"device": dev}})
    print("policy chunk %d, replaying at n_action_steps=%d" % (cfg.chunk_size, a.n_action_steps))
    print()

    pred = np.zeros_like(sent)
    i = 0
    with torch.no_grad():
        while i < n:
            s = ds[i]
            b = {k: s[k][None] for k in
                 ("observation.state", "observation.images.top", "observation.images.wrist")}
            policy.reset()
            o = pre(b)
            m = min(a.n_action_steps, n - i)
            for j in range(m):
                pred[i + j] = post(policy.select_action(o))[0].cpu().numpy()
            i += m

    d = pred - sent
    print("=== replayed prediction vs what was sent ===")
    print("%-9s %10s %10s %10s %10s" % ("joint", "mean diff", "sd", "max |diff|", "sent range"))
    for k, nm in enumerate(J):
        print("%-9s %10.3f %10.3f %10.3f %10.2f"
              % (nm, d[:, k].mean(), d[:, k].std(), np.abs(d[:, k]).max(),
                 sent[:, k].max() - sent[:, k].min()))
    print()
    big = np.abs(d).max()
    if big < 1.0:
        print("Identical to within %.3f units: nothing between the model and the servos" % big)
        print("is altering the action. The bias is in the model itself.")
    else:
        print("Divergence up to %.2f units -- something is changing the action on the way out." % big)

    print()
    print("=== for reference: how far the arm lagged the command ===")
    e = sent - state
    print("%-9s %10s %10s" % ("joint", "mean", "max"))
    for k, nm in enumerate(J):
        print("%-9s %10.2f %10.2f" % (nm, e[:, k].mean(), np.abs(e[:, k]).max()))


if __name__ == "__main__":
    main()
