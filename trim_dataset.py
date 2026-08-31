#!/usr/bin/env python3
"""Rebuild the dataset with the dead time at the head of each episode removed.

WHY
    Measured over the 30 recorded episodes, every one begins with the operator
    getting ready before touching the leader arm:

        frames before motion exceeds 3 units:
            min 67 (2.2 s)   median 100 (3.3 s)   max 250 (8.3 s)
        22.3% of the whole dataset is the arm sitting still

    That idle pose therefore appears paired with two incompatible labels:
    "keep waiting" in most frames and "start reaching" in a few. Nothing in the
    image distinguishes them -- the difference is in the operator's head. ACT
    minimises L1, so it predicts the average, and averaging a wait against a
    reach gives almost no motion.

    Consequences seen on the robot:
      n_action_steps=100 -> commits to a whole chunk, so it does move, but the
                            chunk was planned from an ambiguous state
      n_action_steps=5   -> re-makes the same "wait" decision every 167 ms and
                            never escapes -- the arm barely moves at all

    Trimming the lead-in removes the ambiguity: the first frame of every
    episode is now a state from which the demonstrated answer is "move".

WHY REBUILD RATHER THAN EDIT METADATA IN PLACE
    The v3 layout ties episodes to shared video files through
    from_timestamp/to_timestamp plus dataset_from_index/dataset_to_index, and
    carries per-episode statistics used for normalisation. Hand-editing all of
    that is easy to get subtly wrong and the corruption would be silent.
    Rebuilding through the official API costs one re-encode and keeps every
    invariant lerobot maintains.

    KEEP_LEAD frames of the run-up are retained on purpose, so the policy still
    sees a little of the approach rather than starting mid-motion.

ASCII output only.

Usage:
    python trim_dataset.py --src datasets/so101_pickplace --dst datasets/so101_pickplace_trim
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

THRESH = 3.0        # units of joint motion that count as "started"
KEEP_LEAD = 5       # frames of run-up to keep before the onset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/so101_pickplace")
    ap.add_argument("--dst", default="datasets/so101_pickplace_trim")
    ap.add_argument("--repo-id", default="local/so101_pickplace_trim")
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--keep-lead", type=int, default=KEEP_LEAD)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset("local/src", root=a.src)
    meta = src.meta
    print("source: %d episodes  %d frames  %d fps" % (meta.total_episodes, meta.total_frames, meta.fps))

    dst_path = Path(a.dst)
    if dst_path.exists():
        if not a.force:
            print("[X] %s already exists (use --force to replace)" % dst_path)
            return 1
        shutil.rmtree(dst_path)

    # episode boundaries, straight from the episode metadata
    import glob

    import pandas as pd
    edf = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(f"{a.src}/meta/episodes/**/*.parquet", recursive=True))])
    edf = edf.sort_values("episode_index")
    bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))

    ddf = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(f"{a.src}/data/**/*.parquet", recursive=True))])
    ddf = ddf.sort_values(["episode_index", "frame_index"])
    act_all = np.stack(ddf["action"].to_numpy())

    dst = LeRobotDataset.create(
        repo_id=a.repo_id,
        fps=meta.fps,
        features=meta.features,
        root=a.dst,
        robot_type=meta.robot_type,
        use_videos=True,
    )

    kept_total = dropped_total = 0
    for e, (lo, hi) in enumerate(bounds):
        act = act_all[lo:hi]
        d = np.abs(act - act[0]).max(axis=1)
        onset = int(np.argmax(d > a.thresh)) if (d > a.thresh).any() else 0
        start = max(0, onset - a.keep_lead)
        dropped_total += start
        n = 0
        for i in range(lo + start, hi):
            s = src[i]
            frame = {"task": s["task"]}
            for key in meta.features:
                if key in ("index", "frame_index", "episode_index", "timestamp", "task_index"):
                    continue
                v = s[key]
                if key.startswith("observation.images."):
                    # dataset gives float CHW in [0,1]; writer wants uint8 HWC
                    v = (v.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                elif isinstance(v, torch.Tensor):
                    v = v.numpy()
                frame[key] = v
            dst.add_frame(frame)
            n += 1
        dst.save_episode()
        kept_total += n
        print("ep %2d  onset %3d  dropped %3d  kept %3d" % (e, onset, start, n))

    dst.finalize()
    print()
    print("kept %d frames, dropped %d (%.1f%%)"
          % (kept_total, dropped_total, 100 * dropped_total / (kept_total + dropped_total)))
    print("written to %s" % a.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
