#!/usr/bin/env python3
"""Sanity-check a freshly recorded take before committing to the full session.

Three things go wrong in ways that are cheap to fix now and expensive later:

  1. Dead time at the head of each episode. Recording starts, the operator gets
     ready, and those seconds teach the policy to sit still. The v1 dataset was
     22% dead frames (README #27).

  2. The leader arm or the operator's hand in the overhead frame. lerobot's
     dataset guidance is explicit that only the follower and the manipulated
     objects should be visible; anything present during teleoperation and
     absent at deployment is a systematic train/test difference. The v1 data
     had exactly this.

  3. Inconsistent episode length. v1 ranged 10.5 to 24.1 s for the same task.
     The same picture paired with very different timings is noise the network
     can only average over.

Prints numbers for 1 and 3, and writes a contact sheet so 2 can be judged by eye.

ASCII output only.

Usage:
    python check_take.py --name so101_v2_test
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

FPS = 30
THRESH = 3.0        # units of motion that count as "started"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="so101_v2_test")
    a = ap.parse_args()
    root = os.path.join("datasets", a.name)
    if not os.path.isdir(root):
        raise SystemExit("no dataset at %s" % root)

    edf = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
    ddf = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"{root}/data/**/*.parquet", recursive=True))]).sort_values(["episode_index", "frame_index"])
    act = np.stack(ddf["action"].to_numpy())
    ep = ddf["episode_index"].to_numpy()

    print("dataset %s : %d episodes" % (a.name, len(edf)))
    print()

    print("%4s %10s %12s %10s" % ("ep", "length s", "dead head s", "verdict"))
    lens, deads = [], []
    for e in np.unique(ep):
        x = act[ep == e]
        d = np.abs(x - x[0]).max(axis=1)
        onset = int(np.argmax(d > THRESH)) if (d > THRESH).any() else len(x)
        L, D = len(x) / FPS, onset / FPS
        lens.append(L)
        deads.append(D)
        flags = []
        if D > 1.5:
            flags.append("dead head")
        if L > 14:
            flags.append("slow")
        print("%4d %10.1f %12.1f %10s" % (e, L, D, ", ".join(flags) if flags else "ok"))
    lens, deads = np.array(lens), np.array(deads)
    print()
    print("length   mean %.1f s   spread %.1f s  (want them close together, 8-10 s)"
          % (lens.mean(), lens.max() - lens.min()))
    print("dead head mean %.1f s   max %.1f s     (want under 0.5 s)"
          % (deads.mean(), deads.max()))
    print("dead frames %.1f%% of the take   (v1 was 22.3%%)"
          % (100 * deads.sum() / lens.sum()))
    print()

    # contact sheet of the overhead view: is the leader arm or a hand in frame?
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/check", root=root)
    starts = edf["dataset_from_index"].astype(int).tolist()
    tiles = []
    for e, s0 in enumerate(starts):
        n = int(edf["length"].iloc[e])
        for frac, tag in ((0.0, "start"), (0.5, "mid")):
            i = s0 + int(frac * (n - 1))
            img = ds[i]["observation.images.top"]
            img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (320, 240))
            lab = "ep%d %s" % (e, tag)
            for c, t in (((0, 0, 0), 4), ((0, 255, 255), 1)):
                cv2.putText(img, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, t)
            tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    out = "check_take_overhead.png"
    cv2.imwrite(out, sheet)
    print("wrote %s -- look for the leader arm or your hand anywhere in frame." % out)
    print("Anything that moves during recording but is absent at deployment has")
    print("to go before the real session.")


if __name__ == "__main__":
    main()
