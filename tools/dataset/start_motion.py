r"""How soon does the arm start moving, in each dataset?

The into-bowl policy acts immediately; the out-of-bowl one waits a long time
before it tries anything. A policy waits because it was shown waiting, so the
place to look is the beginning of the demonstrations rather than the model.

Reports joint speed against position in the episode, on the same data the
training saw.

    python tools/dataset/start_motion.py datasets/so101_v3 datasets/so101_color6
"""

import glob
import sys

import numpy as np
import pandas as pd


def profile(root: str, bins: int = 20):
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {root}/data")
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df.sort_values(["episode_index", "frame_index"])

    rows, still = [], []
    for _, g in df.groupby("episode_index"):
        s = np.stack(g["observation.state"].to_numpy())
        # Speed per frame, summed over joints: how much the arm is moving at all.
        v = np.abs(np.diff(s, axis=0)).sum(axis=1)
        if len(v) < bins:
            continue
        # How long before it first moves in earnest, as a fraction of the episode.
        thr = np.percentile(v, 75) * 0.25
        moving = np.where(v > thr)[0]
        still.append((moving[0] if len(moving) else len(v)) / len(v))
        idx = (np.arange(len(v)) / len(v) * bins).astype(int)
        rows.append([v[idx == b].mean() for b in range(bins)])

    return np.array(rows), np.array(still), len(df)


for root in sys.argv[1:]:
    prof, still, frames = profile(root)
    med = np.median(prof, axis=0)
    print(f"\n{root}   {len(prof)} episodes, {frames} frames")
    print(f"  first real movement at {np.median(still) * 100:.0f}% of the way in "
          f"(p90 {np.percentile(still, 90) * 100:.0f}%)")
    peak = med.max()
    print("  speed through the episode (each block is 5% of it):")
    print("   ", "".join("#" if v > 0.6 * peak else ("+" if v > 0.25 * peak else ".")
                         for v in med))
    print(f"    first 10%: {med[:2].mean() / peak * 100:4.0f}% of peak speed")
