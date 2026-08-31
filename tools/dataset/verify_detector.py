"""Draw the block detector's output so it can be checked by eye.

Several conclusions rested on this detector -- that the block positions span
only 59x112 px, that the live frame sat inside the training region, that pixel
position predicts the reach angle. None of that means anything if the detector
is wrong, and it was never verified.

It is a plain HSV threshold for yellow followed by the largest contour. Nothing
clever, and easy to fool: yellow-ish reflections, the block partly occluded by
the gripper, or shadows can all move or lose it.

Writes a grid of every episode's opening frame with the box drawn, and an
annotated video of one episode so the per-frame behaviour can be scrubbed.

ASCII output only.
"""

import argparse
import glob

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = "datasets/so101_v3"
LO, HI = (18, 90, 90), (38, 255, 255)


def detect(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LO, HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return None
    big = max(c, key=cv2.contourArea)
    return cv2.boundingRect(big), cv2.contourArea(big), len(c)


def to_bgr(chw):
    a = (chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def annotate(bgr, label):
    r = detect(bgr)
    out = bgr.copy()
    if r is None:
        cv2.putText(out, "NOT FOUND", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        (x, y, w, h), area, ncand = r
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (x + w // 2, y + h // 2), 3, (0, 0, 255), -1)
        txt = "%dx%d a=%d n=%d" % (w, h, area, ncand)
        cv2.putText(out, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(out, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    for c_, t_ in (((0, 0, 0), 4), ((0, 255, 255), 1)):
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_, t_)
    return out, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=5, help="which episode to render as video")
    a = ap.parse_args()

    ds = LeRobotDataset("local/v2", root=ROOT)
    edf = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
    starts = edf["dataset_from_index"].astype(int).tolist()
    lens = edf["length"].astype(int).tolist()

    # grid of every episode's opening frame
    tiles, missing, multi = [], 0, 0
    for e, i in enumerate(starts):
        img, r = annotate(to_bgr(ds[i]["observation.images.top"]), "ep%d" % e)
        if r is None:
            missing += 1
        elif r[2] > 1:
            multi += 1
        tiles.append(cv2.resize(img, (256, 192)))
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)])
    cv2.imwrite("detector_grid.png", grid)
    print("episode-start frames: %d" % len(starts))
    print("  not found        : %d" % missing)
    print("  >1 yellow blob   : %d   (the largest is taken, which may be the wrong one)" % multi)
    print("wrote detector_grid.png")

    # annotated video of one episode, every frame
    e = a.episode
    lo, n = starts[e], lens[e]
    vw = cv2.VideoWriter("detector_ep%d.mp4" % e, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    miss = 0
    for k in range(n):
        img, r = annotate(to_bgr(ds[lo + k]["observation.images.top"]),
                          "ep%d  frame %d/%d" % (e, k, n))
        if r is None:
            miss += 1
        vw.write(img)
    vw.release()
    print()
    print("episode %d: %d frames, block missed in %d (%.1f%%)" % (e, n, miss, 100 * miss / n))
    print("wrote detector_ep%d.mp4" % e)


if __name__ == "__main__":
    main()
