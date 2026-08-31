#!/usr/bin/env python3
"""Find the grasp by watching the block move, not by reading the gripper.

Every quantitative result so far has leaned on a landmark for "the moment of the
grasp", and every version of it has been unreliable:

  * minimum shoulder_lift -- lands on the home pose in a rollout, since home is
    the lowest the arm ever sits
  * gripper closing after its widest opening -- rendering the wrist view at the
    frames it chose showed the block properly framed in only about half of them

Regressions fitted on a landmark that is right half the time are fitting noise,
which is why "pixel position predicts the grasp pose" kept coming out weak.

The block itself is a better witness. It sits still until it is picked up, so
the first frame where its detected position departs from where it started is
the grasp -- and that same fact doubles as an objective success test, which we
have never had: the block moved, or it did not.

Exports `grasp_frame(...)` for other scripts, and when run directly renders the
chosen frames so the landmark can be checked by eye rather than trusted.

ASCII output only.
"""

import argparse
import glob

import cv2
import numpy as np
import pandas as pd

HSV_LO, HSV_HI = (18, 90, 90), (38, 255, 255)


def block_xy(chw):
    """Centre of the largest yellow blob, or None."""
    a = (chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    hsv = cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, HSV_LO, HSV_HI),
                         cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return None
    x, y, w, h = cv2.boundingRect(max(c, key=cv2.contourArea))
    return np.array([x + w / 2.0, y + h / 2.0])


def episode_video(root, edf, e, cam="top"):
    """Path and start time of one episode inside the shared video file."""
    row = edf.iloc[e]
    pre = "videos/observation.images.%s/" % cam
    fi = int(row[pre + "file_index"])
    ci = int(row[pre + "chunk_index"])
    path = "%s/videos/observation.images.%s/chunk-%03d/file-%03d.mp4" % (root, cam, ci, fi)
    return path, float(row[pre + "from_timestamp"]), float(row[pre + "to_timestamp"])


def track(root, edf, e, step=2, cam="top"):
    """Block position through an episode, read straight from the mp4.

    Random per-frame access through LeRobotDataset opens a fresh AV1 decoder for
    every request and exhausts memory partway through the dataset
    (`avcodec_open2(libdav1d)` fails with ENOMEM). Episodes are stored
    contiguously inside one file, so seeking once and reading forward is both
    reliable and far faster.
    """
    path, t0, t1 = episode_video(root, edf, e, cam)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    out, k = [], 0
    n = int(round((t1 - t0) * fps))
    while k < n:
        ok, fr = cap.read()
        if not ok:
            break
        if k % step == 0:
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            m = cv2.morphologyEx(cv2.inRange(hsv, HSV_LO, HSV_HI),
                                 cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if c:
                x, y, w, h = cv2.boundingRect(max(c, key=cv2.contourArea))
                out.append((k, np.array([x + w / 2.0, y + h / 2.0])))
        k += 1
    cap.release()
    return out


def grasp_frame(root, edf, e, move_px=25.0, step=2):
    """Grasp = the moment the block stops moving in the WRIST view.

    The wrist camera is the better witness. Once the block is held it travels
    with the gripper, so it goes still in that view, while during the approach
    it sweeps across it. The grasp is that transition, and it is a sharper
    signal than any threshold on distance. The block also fills a large part of
    the wrist frame at that range, so the detection is far more reliable than
    the ~34 px it occupies overhead.

    The overhead view is kept as corroboration: it says whether the block
    actually left its resting place. Without that, a gripper that stopped just
    short of the block would look identical -- nothing moving in either view.

    Returns (frame_offset, resting_xy_overhead, moved).
    """
    # --- corroboration: did the block leave its place on the table? ---
    top = track(root, edf, e, step, cam="top")
    if len(top) < 8:
        return None, None, False
    early = np.array([p for k, p in top if k < 30]) if any(k < 30 for k, _ in top)         else np.array([top[0][1]])
    home = np.median(early, axis=0)
    left = max(np.linalg.norm(p - home) for _, p in top) > move_px
    if not left:
        return None, home, False

    # --- primary: where does it go still in the wrist view? ---
    wr = track(root, edf, e, step, cam="wrist")
    if len(wr) < 8:
        return None, home, True

    ks = np.array([k for k, _ in wr])
    xy = np.array([p for _, p in wr])
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1)        # px per sampled frame
    fast = max(6.0, float(np.percentile(v, 75)))           # what "approaching" looks like
    hold = max(3, int(0.6 * 30 / step))                    # must stay still ~0.6 s

    for i in range(len(v) - hold):
        if v[i] >= fast:
            continue
        # The approach must already have happened. Looking back only a fraction
        # of a second found the home pose instead: the arm sits still there too,
        # and at 0.3 s there is no history to look back at.
        if ks[i] < 2.0 * 30:
            continue
        if np.all(v[i:i + hold] < fast * 0.5) and np.any(v[:i] >= fast):
            return int(ks[i]), home, True

    # never settled: fall back to the overhead departure
    d = np.array([np.linalg.norm(p - home) for _, p in top])
    return int(np.array([k for k, _ in top])[int(np.argmax(d))]), home, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/so101_v3")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--move-px", type=float, default=25.0)
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/x", root=a.root)
    edf = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"{a.root}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
    bounds = list(zip(edf["dataset_from_index"].astype(int), edf["dataset_to_index"].astype(int)))

    print("%4s %10s %10s %10s" % ("ep", "grasp @s", "moved", "length s"))
    tiles, ok, n = [], 0, 0
    for e, (lo, hi) in enumerate(bounds):
        k, home, moved = grasp_frame(a.root, edf, e, a.move_px)
        n += 1
        if moved:
            ok += 1
        if e < a.episodes:
            print("%4d %10s %10s %10.1f"
                  % (e, "%.1f" % (k / 30) if k is not None else "-", moved, (hi - lo) / 30))
            idx = lo + (k if k is not None else 0)
            img = (ds[idx]["observation.images.wrist"].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (320, 240))
            lab = "ep%d %s" % (e, "@%.1fs" % (k / 30) if k is not None else "NO MOVE")
            for c_, t_ in (((0, 0, 0), 4), ((0, 255, 255), 1)):
                cv2.putText(img, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_, t_)
            tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite("grasp_landmark.png",
                np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]))
    print()
    print("episodes where the block moved: %d / %d" % (ok, n))
    print("wrote grasp_landmark.png -- wrist view at each chosen frame;")
    print("the block should be between the fingers in every one.")


if __name__ == "__main__":
    main()
