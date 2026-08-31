#!/usr/bin/env python3
"""Diagnose a recorded rollout against the training data.

Every check I ran before this fed the policy observations taken from the
training set, so they could only ever prove things about the model on paper.
They could not see what happens on the robot. This reads a rollout recorded
with `robot.ps1 eval -Record` and asks three questions whose answers point at
different layers and cannot both be true:

  Q1  Was the observation in distribution?
      Nearest-neighbour image similarity against training frames, compared
      with how similar training frames are to each other. Below that baseline
      means the policy was looking at something it was never trained on --
      a perception problem, not a policy problem.

  Q2  Did the online actions match what the policy predicts offline from the
      SAME recorded observations?
      Same input, same weights. A mismatch means something in the deployment
      path (normalisation, device, clamping, action queue) alters the command.
      A match means the model genuinely wants to do this, and the fault is in
      what it learned.

  Q3  What is the oscillation period?
      If the shaking lines up with the replanning interval (n_action_steps
      frames) then the policy changes its mind at every replan -- it is not
      converging on a plan. If it is faster than that, it is inside a single
      chunk and the chunk itself is oscillatory.

ASCII output only.

Usage:
    python analyze_rollout.py [--rollout datasets/rollout_probe] [--n-action-steps 5]
"""

import argparse
import glob

import numpy as np
import pandas as pd
import torch

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def sig_batch(imgs):
    """32x32 mean-removed unit-norm signatures from CHW float tensors in [0,1]."""
    out = []
    for t in imgs:
        g = t.mean(0)                                   # grayscale
        g = torch.nn.functional.interpolate(
            g[None, None], size=(32, 32), mode="area")[0, 0].numpy().astype(np.float64)
        v = g - g.mean()
        n = np.linalg.norm(v)
        out.append(v.ravel() / n if n > 1e-9 else v.ravel())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", default="datasets/rollout_probe")
    ap.add_argument("--train", default="datasets/so101_v3")
    ap.add_argument("--ckpt", default="policies/act_v3_100k")
    ap.add_argument("--n-action-steps", type=int, default=5)
    ap.add_argument("--samples", type=int, default=40)
    a = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    roll = LeRobotDataset("local/roll", root=a.rollout)
    train = LeRobotDataset("local/train", root=a.train)
    print("rollout  %d frames" % len(roll))
    print("training %d frames" % len(train))
    print()

    idx = np.linspace(0, len(roll) - 1, min(a.samples, len(roll))).astype(int)
    tidx = np.linspace(0, len(train) - 1, 120).astype(int)

    # ---------- Q1 ----------
    print("=== Q1  was the observation in distribution? ===")
    for cam in ("top", "wrist"):
        key = f"observation.images.{cam}"
        R = sig_batch([roll[int(i)][key] for i in idx])
        T = sig_batch([train[int(i)][key] for i in tidx])
        nn = (R @ T.T).max(axis=1)
        C = T @ T.T
        np.fill_diagonal(C, -1)
        base = C.max(axis=1)
        verdict = "IN distribution" if nn.mean() >= base.mean() - 0.05 else "OUT of distribution"
        print("  %-6s rollout-vs-train %.3f   train-vs-train %.3f   -> %s"
              % (cam, nn.mean(), base.mean(), verdict))
    print()

    # ---------- Q2 ----------
    print("=== Q2  do online actions match offline prediction? ===")
    policy = ACTPolicy.from_pretrained(a.ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(dev).eval()
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=a.ckpt,
        preprocessor_overrides={"device_processor": {"device": dev}})

    errs = []
    with torch.no_grad():
        for i in idx:
            s = roll[int(i)]
            b = {k: s[k][None] for k in
                 ("observation.state", "observation.images.top", "observation.images.wrist")}
            policy.reset()
            pred = post(policy.select_action(pre(b)))[0].cpu().numpy()
            errs.append(np.abs(pred - s["action"].numpy()))
    errs = np.array(errs)
    print("  %-15s %10s" % ("joint", "MAE"))
    for j, nm in enumerate(JOINTS):
        print("  %-15s %10.2f" % (nm, errs[:, j].mean()))
    tot = errs.mean()
    print("  total %.2f" % tot)
    print("  -> %s" % ("MATCHES (fault is in what the model learned)" if tot < 5
                       else "MISMATCH (fault is in the deployment path)"))
    print()

    # ---------- Q3 ----------
    print("=== Q3  what is the oscillation period? ===")
    files = sorted(glob.glob(f"{a.rollout}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("frame_index")
    act = np.stack(df["action"].to_numpy())
    fps = roll.fps
    print("  replan interval = %d frames (%.0f ms)" % (a.n_action_steps, a.n_action_steps / fps * 1000))
    print("  %-15s %10s %12s %12s" % ("joint", "std", "reversals/s", "period(fr)"))
    for j, nm in enumerate(JOINTS):
        c = act[:, j]
        d = np.diff(c)
        rev = np.sum(np.sign(d[1:]) != np.sign(d[:-1]))
        rev_s = rev / (len(c) / fps)
        period = 2 * len(c) / rev if rev else float("inf")
        print("  %-15s %10.2f %12.1f %12.1f" % (nm, c.std(), rev_s, period))
    print()
    print("  period close to %d frames -> it changes its mind at every replan" % a.n_action_steps)
    print("  period much smaller     -> the chunk itself oscillates")


if __name__ == "__main__":
    main()
