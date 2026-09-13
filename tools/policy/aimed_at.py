r"""Which cube was it aiming at? Read it off the wrist view when the gripper shuts.

An earlier attempt read the outcome instead -- which cube ended up moved -- and
that is the wrong question. It cannot tell a wrong choice from a right choice
that missed, and missing is most of what happens. Intent lives in the motion.

At the moment the gripper closes, the wrist camera is looking straight at
whatever the arm decided to take, hit or miss. So: find that moment from the
gripper trace, detect the cubes in that frame, and report which colour lies
closest to the point between the fingers.

The gripper opens and closes twice in a demonstration, once to grasp and once to
release onto the table, so the first cycle is the one that matters.

Two things keep it honest. The frames it chose are written out to be looked at,
because every previous landmark here was believed and then turned out to be
right about half the time. And it is scored on demonstrations first, where the
named cube is the answer.

    .\.venv-win\Scripts\python.exe tools\policy\aimed_at.py [dataset] [-n N] [-out DIR]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
import box_cubes as B                                  # noqa: E402

HOLD = 6                # frames a closure must hold: 0.2 s at 30 fps
# Where the fingers meet in the wrist view, as a fraction of width and height.
# Not a guess: measured below from the frames themselves and printed, so a
# changed camera mount shows up instead of silently biasing every episode.
GRIP_XY = (0.50, 0.72)


def grasp_frame(gripper: np.ndarray) -> int | None:
    """First sustained closure after the first real opening."""
    top = float(gripper.max())
    if top < 5:
        return None
    opened = np.flatnonzero(gripper > 0.5 * top)
    if not len(opened):
        return None
    i = int(opened[0])
    peak = float(gripper[i:].max())
    for j in range(i + 1, len(gripper) - HOLD):
        if np.all(gripper[j:j + HOLD] < 0.4 * peak):
            return j
    return None


def main(argv: list[str]) -> int:
    name = next((a for a in argv if not a.startswith("-")), "so101_color6")
    n = int(argv[argv.index("-n") + 1]) if "-n" in argv else 20
    out = Path(argv[argv.index("-out") + 1]) if "-out" in argv else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    ds = ROOT / "datasets" / name
    B.DS = ds
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    anames = info["features"]["action"]["names"]
    gcol = anames.index("gripper.pos")
    ep = pd.concat([pd.read_parquet(f) for f in
                    sorted((ds / "meta" / "episodes").rglob("*.parquet"))],
                   ignore_index=True).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in
                      sorted((ds / "data").rglob("*.parquet"))], ignore_index=True)
    tasks = pd.read_parquet(ds / "meta" / "tasks.parquet")
    task_of = {int(v): k for k, v in tasks["task_index"].items()}
    cache = {int(e): r.sort_values("frame_index") for e, r in data.groupby("episode_index")}
    chosen = sorted(cache)[::max(1, len(cache) // n)][:n]

    hit = miss = blank = 0
    print(f"{ds.name}: {len(chosen)} episodes, wrist view at the first closure\n")
    print(f"  {'ep':>4}  {'asked for':<9}{'aimed at':<9}{'at':>5}  {'px':>5}  others")
    for e in chosen:
        r = cache[e]
        g = np.stack(r["action"].to_numpy())[:, gcol]
        j = grasp_frame(g)
        named = task_of[int(r["task_index"].iloc[0])].split()[2]
        if j is None:
            blank += 1
            print(f"  {e:>4}  {named:<9}{'-':<9}{'-':>5}")
            continue
        t = float(r["timestamp"].to_numpy()[j])
        rgb = B.frame_at(info, ep.loc[e], "observation.images.wrist", t)
        h, w = rgb.shape[:2]
        anchor = np.array([GRIP_XY[0] * w, GRIP_XY[1] * h])
        found = B.find(rgb, B.WRIST_AREA)
        if not found:
            blank += 1
            print(f"  {e:>4}  {named:<9}{'none seen':<9}{j:>5}")
            continue
        best = {}
        for d in found:
            dist = float(np.linalg.norm(np.asarray(d["centre"]) - anchor))
            if d["colour"] not in best or dist < best[d["colour"]]:
                best[d["colour"]] = dist
        order = sorted(best, key=best.get)
        aimed = order[0]
        # How far is "at"? One cube width, measured in this frame rather than
        # fixed, because the wrist camera's distance to the cubes varies. Two
        # demonstrations picked a frame over the empty table with a cube in the
        # far corner 400 px away, and reporting that as an aim would be luck.
        scale = float(np.median([d["box"][2] for d in found]))
        if best[aimed] > scale:
            blank += 1
            print(f"  {e:>4}  {named:<9}{'unclear':<9}{j:>5}  {best[aimed]:5.0f}"
                  f"   nearest cube is {best[aimed] / scale:.1f} cube widths away")
            continue
        if aimed == named:
            hit += 1
        else:
            miss += 1
        others = " ".join(f"{c}:{best[c]:.0f}" for c in order[1:4])
        print(f"  {e:>4}  {named:<9}{aimed:<9}{j:>5}  {best[aimed]:5.0f}  {others}"
              + ("" if aimed == named else "   <-- different cube"))
        if out:
            img = B.draw(rgb, found, named)
            cv2.drawMarker(img, (int(anchor[0]), int(anchor[1])), (255, 255, 255),
                           cv2.MARKER_CROSS, 26, 2)
            cv2.imwrite(str(out / f"ep{e:03d}_{named}_aimed_{aimed}.png"), img)

    total = hit + miss + blank
    print(f"\n  aimed at the named cube  {hit:>3}/{total}  ({hit / total:.0%})")
    print(f"  aimed at another one     {miss:>3}/{total}")
    print(f"  no reading               {blank:>3}/{total}")
    if out:
        print(f"\n  frames written to {out} -- look at them before believing this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
