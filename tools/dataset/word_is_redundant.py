r"""Is the colour word ever the only thing that says which cube to take?

The policy ignores the colour word entirely: handed the wrong one it plans the
same move (tools/policy/knows_which_cube.py). Training cannot reward reading a
word that is never needed, so the question is whether the data ever needs it.

At each point through the episode this measures, with no model involved, how
well the arm's own joint angles already identify that episode's next 50 steps --
nearest neighbours in joint space, the episode itself excluded, scored the same
way as the policy. Where that is already high, the word is redundant: the
demonstrator has committed and the arm's pose says where it is going.

    .\.venv-win\Scripts\python.exe tools\dataset\word_is_redundant.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets" / "so101_color6"
HORIZON = 50

data = pd.concat([pd.read_parquet(f) for f in
                  sorted((DS / "data").rglob("*.parquet"))], ignore_index=True)
cache = {int(e): r.sort_values("frame_index")
         for e, r in data.groupby("episode_index")}
episodes = sorted(cache)
colour = np.array([int(cache[e]["task_index"].iloc[0]) for e in episodes])

print(f"{len(episodes)} episodes, {len(set(colour))} colours, "
      f"{HORIZON}-step futures")
print()
print("  point in episode      5%   10%   20%   30%   40%   50%   60%   70%")
rows = {"futures disagree by": [], "joint angles alone": [],
        "colour label alone": []}

for frac in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
    keep, F, S = [], [], []
    for e in episodes:
        r = cache[e]
        i = int(round(frac * len(r)))
        if i + HORIZON > len(r):
            continue
        keep.append(e)
        F.append(np.stack(r["action"].to_numpy()[i:i + HORIZON]))
        S.append(np.asarray(r["observation.state"].to_numpy()[i], dtype=float))
    F, S = np.stack(F), np.stack(S)
    c = colour[[episodes.index(e) for e in keep]]
    rows["futures disagree by"].append(float(np.mean(np.std(F, axis=0))))

    def rank(pred):
        out = []
        for k in range(len(F)):
            d = np.sqrt(((F - pred[k][None]) ** 2).mean(axis=(1, 2)))
            out.append(float((np.delete(d, k) > d[k]).mean()))
        return float(np.mean(out))

    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    rows["joint angles alone"].append(rank(F[np.argsort(d2, axis=1)[:, :3]]
                                           .mean(axis=1)))

    # Knowing only the colour, and nothing else: the average future of the other
    # episodes that share it. This is the most the word alone can buy.
    same = np.stack([F[(c == c[k]) & (np.arange(len(F)) != k)].mean(axis=0)
                     for k in range(len(F))])
    rows["colour label alone"].append(rank(same))

for label, vals in rows.items():
    fmt = "{:5.1f}" if label.startswith("futures") else "{:5.1%}"
    print(f"  {label:<20}" + " ".join(fmt.format(v) for v in vals))

print()
print("Chance is 50%. Where the joint angles alone already score high, the word")
print("adds nothing a policy could be graded on.")
