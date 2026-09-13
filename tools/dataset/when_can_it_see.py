r"""When the choice has to be made, can the wrist camera see anything to choose?

The wrist view centred on the target is what makes the grasp look obvious, but
it is a consequence of the operator having already chosen, not the cause. This
asks when in the episode each camera can actually see the cubes at all.

Detection is deliberately crude and colour-blind: the table is white, the bowl
is black, the gripper is black, and only the cubes are strongly coloured, so the
fraction of strongly saturated pixels says whether cubes are in frame. Nothing
here depends on telling one colour from another.

    .\.venv-win\Scripts\python.exe tools\dataset\when_can_it_see.py
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets" / "so101_color6"
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
EPISODES = int(os.environ.get("EPISODES", 60))
SAT, VAL = 90, 60          # a cube face, not white table and not black bowl


def coloured(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(((hsv[..., 1] > SAT) & (hsv[..., 2] > VAL)).mean())


info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
ep = pd.concat([pd.read_parquet(f) for f in
                sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
               ignore_index=True).set_index("episode_index")
data = pd.concat([pd.read_parquet(f) for f in
                  sorted((DS / "data").rglob("*.parquet"))], ignore_index=True)
lengths = {int(e): len(r) for e, r in data.groupby("episode_index")}
stamps = {int(e): r.sort_values("frame_index")["timestamp"].to_numpy()
          for e, r in data.groupby("episode_index")}

chosen = sorted(lengths)[::max(1, len(lengths) // EPISODES)][:EPISODES]

import av

frac_of = {}
for key, cam in (("observation.images.top", "top"),
                 ("observation.images.wrist", "wrist")):
    per_frac = {f: [] for f in FRACS}
    for e in chosen:
        row = ep.loc[e]
        v = DS / info["video_path"].format(
            video_key=key,
            chunk_index=int(row[f"videos/{key}/chunk_index"]),
            file_index=int(row[f"videos/{key}/file_index"]))
        base = float(row[f"videos/{key}/from_timestamp"])
        with av.open(str(v)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for f in FRACS:
                i = min(int(round(f * lengths[e])), lengths[e] - 1)
                t = base + float(stamps[e][i])
                c.seek(int(max(t - 0.5, 0) / s.time_base), stream=s)
                for pic in c.decode(video=0):
                    if float(pic.pts * s.time_base) >= t - 1e-6:
                        per_frac[f].append(coloured(pic.to_ndarray(format="rgb24")))
                        break
    frac_of[cam] = per_frac

print(f"{len(chosen)} episodes")
print("\nshare of the frame that is a coloured cube, median over episodes")
print("  camera   " + " ".join(f"{f:5.0%}" for f in FRACS))
for cam in ("top", "wrist"):
    print(f"  {cam:<8} " + " ".join(f"{np.median(frac_of[cam][f]):5.1%}"
                                    for f in FRACS))

# Blind means the cube pixels are a smaller share than the darkest tenth of
# frames that do contain the bowl, i.e. nothing worth choosing between.
floor = float(np.quantile(frac_of["top"][0.0], 0.10)) / 4
print(f"\nepisodes where the view is effectively empty (< {floor:.1%} coloured)")
print("  camera   " + " ".join(f"{f:5.0%}" for f in FRACS))
for cam in ("top", "wrist"):
    print(f"  {cam:<8} " + " ".join(
        f"{np.mean(np.array(frac_of[cam][f]) < floor):5.0%}" for f in FRACS))

print("\nThe cube has to be chosen before the arm moves. A camera that is empty")
print("then cannot be what chooses it.")
