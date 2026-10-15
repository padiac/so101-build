r"""Per-episode rubric and motion metrics, computed the same way for demonstrations
and for recorded rollouts, so the two can be put side by side.

Each column answers one item of a staged rubric (Kress-Gazit et al. 2024; TRI LBM
2025) or one failure category (Yu & Qiu 2026, SO-101 benchmark), from the measured
joint trace alone. Nothing here looks at the camera; aimed_at.py does that part.

  onset_s        seconds until the arm starts to move           rubric: started
  closes         gripper closures in the episode                repetition loop if > 2
  first_hold     did the first closure stop on something        rubric: grasped
                 (measured gripper stays wider than the command: a cube is in the way)
  holds          closures that stopped on something
  reach_s        onset to first closure
  sparc_reach    SPARC smoothness of the reach, onset -> first closure
  sparc_all      SPARC of the whole motion, onset -> end
  reversals_hz   direction reversals of the arm joints per second of motion
  home_returns   times the arm went out and came back near its start pose
                 before the first held grasp -- the "tries, misses, retreats" loop

SPARC is the spectral arc length of the speed profile (Balasubramanian et al. 2015,
"On the analysis of movement smoothness"), the smoothness measure NVIDIA's RoboLab
reports. It is always negative; closer to zero is smoother. It depends on duration,
so compare segments of the same kind.

Closures are found on the COMMANDED gripper and judged on the MEASURED one. A
closure on a cube is commanded to about 8 and stalls at about 18, because the cube
blocks the fingers; an empty closure is commanded to 0 and reaches 0-1. The gap
threshold is printed from the dataset, because a demonstration's first closure
always lands on a cube.

    .\.venv-win\Scripts\python.exe tools\rollout\rubric_metrics.py datasets\so101_color6 -out logs\rubric\demo.csv
    .\.venv-win\Scripts\python.exe tools\rollout\rubric_metrics.py datasets\rollouts\run01 -out logs\rubric\run01.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30
ARM = slice(0, 5)
GRIP = 5
MOVE_THRESH = 0.15      # per-frame summed arm motion that counts as moving
HOLD = 6                # frames a closure must last
HOME_RADIUS = 12.0      # joint-space distance that counts as "back near start"
OUT_RADIUS = 30.0       # and as "went out"


def sparc(speed: np.ndarray, fs: float = FPS, padlevel: int = 4,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length, reference implementation of Balasubramanian et al."""
    if len(speed) < 8 or np.allclose(speed, 0):
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    idx = np.flatnonzero(mag >= amp_th)
    f, mag = f[idx[0]:idx[-1] + 1], mag[idx[0]:idx[-1] + 1]
    if len(f) < 2:
        return float("nan")
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(mag) ** 2)))


def closures(cmd: np.ndarray, meas: np.ndarray, drop: float = 8.0):
    """(start, end, gap) of each sustained closure on the commanded gripper.

    Hysteresis both ways: a closure starts when the command falls `drop` below
    its peak since the last closure, and ends when it rises `drop` above its trough
    within this closure. Measuring the end against the trough rather than the old
    peak matters: a release that opens from 7 to 29 after an opening of 38 would
    otherwise merge grasp, release and homing into one closure.

    gap = median of measured minus commanded over the settled part of the closure
    (command within 3 of its trough): large when something is in the fingers.
    """
    out, n = [], len(cmd)
    peak, i = float(cmd[0]), 0
    while i < n:
        peak = max(peak, float(cmd[i]))
        if cmd[i] > peak - drop:
            i += 1
            continue
        start, trough, j = i, float(cmd[i]), i
        while j < n and cmd[j] < trough + drop:
            trough = min(trough, float(cmd[j]))
            j += 1
        if j - start >= HOLD:
            seg = slice(start, j)
            settled = cmd[seg] <= trough + 3
            gap = float(np.median((meas[seg] - cmd[seg])[settled]))
            out.append((start, j, gap))
        peak = float(cmd[j]) if j < n else trough
        i = j
    return out


def episode_metrics(state: np.ndarray, action: np.ndarray, hold_thresh: float) -> dict:
    arm = state[:, ARM]
    step = np.abs(np.diff(arm, axis=0)).sum(1)
    moving = np.flatnonzero(step > MOVE_THRESH)
    onset = int(moving[0]) if len(moving) else len(state) - 1
    speed = np.linalg.norm(np.diff(arm, axis=0), axis=1) * FPS
    cl = closures(action[:, GRIP], state[:, GRIP])
    first = cl[0] if cl else None
    held = [c for c in cl if c[2] > hold_thresh]

    reach_end = first[0] if first else len(state) - 1
    d = np.sign(np.diff(arm[onset:], axis=0))
    d = d[np.abs(np.diff(arm[onset:], axis=0)).sum(1) > MOVE_THRESH]
    flips = int(np.sum((d[1:] * d[:-1]) < 0)) if len(d) > 1 else 0
    motion_s = max((len(state) - onset) / FPS, 1e-6)

    start = arm[onset]
    dist = np.linalg.norm(arm - start, axis=1)
    stop = held[0][0] if held else len(state)
    returns, out = 0, False
    for x in dist[onset:stop]:
        if not out and x > OUT_RADIUS:
            out = True
        elif out and x < HOME_RADIUS:
            returns += 1
            out = False

    return dict(
        frames=len(state),
        onset_s=onset / FPS,
        closes=len(cl),
        first_hold=bool(first and first[2] > hold_thresh),
        holds=len(held),
        reach_s=(reach_end - onset) / FPS if first else float("nan"),
        sparc_reach=sparc(speed[onset:reach_end]) if reach_end - onset > 8 else float("nan"),
        sparc_all=sparc(speed[onset:]),
        reversals_hz=flips / motion_s,
        home_returns=returns,
    )


def load(root: Path):
    data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                      for f in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
    eps = {}
    for e, g in data.groupby("episode_index"):
        g = g.sort_values("frame_index")
        eps[int(e)] = (np.stack(g["observation.state"].to_numpy()), np.stack(g["action"].to_numpy()))
    return eps


def hold_threshold(eps: dict):
    """Separate empty closures from closures on a cube, read from demonstrations.

    Empty: each demonstration's LAST closure, the return to the start pose with
    nothing in the fingers. The measured-minus-command gap of those sits near 0.5;
    closures on a cube sit at 3 and above, and 1-3 is nearly empty (a gentle
    squeeze gives 3, a firm one 10+). The threshold is the 99th percentile of the
    empty gaps, floored at 1.5 so a noisy servo cannot pull it to zero.
    """
    empty, first = [], []
    for st, ac in eps.values():
        c = closures(ac[:, GRIP], st[:, GRIP])
        if len(c) >= 2:
            empty.append(c[-1][2])
            first.append(max(x[2] for x in c[:-1]))
    thr = max(float(np.percentile(empty, 99)), 1.5)
    return thr, float(np.median(empty)), float(np.median(first))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("-out", type=Path)
    ap.add_argument("--hold-thresh", type=float, help="reuse the threshold measured on demonstrations")
    a = ap.parse_args()

    eps = load(a.root)
    if a.hold_thresh is None:
        thr, empty, cube = hold_threshold(eps)
        print(f"gripper gap (measured - command): empty close {empty:.1f}, "
              f"close on a cube {cube:.1f} (median) -> held if > {thr:.1f}")
    else:
        thr = a.hold_thresh
        print(f"gripper: held if > {thr:.1f} (given)")

    rows = [dict(episode=e, **episode_metrics(st, ac, thr)) for e, (st, ac) in sorted(eps.items())]
    df = pd.DataFrame(rows)
    info = json.loads((a.root / "meta" / "info.json").read_text(encoding="utf-8"))
    print(f"{a.root.name}: {len(df)} episodes @ {info['fps']} fps\n")
    cols = ["onset_s", "reach_s", "closes", "holds", "sparc_reach", "sparc_all", "reversals_hz", "home_returns"]
    print(df[cols].describe(percentiles=[.1, .5, .9]).round(2).to_string())
    print(f"\nfirst closure held something: {df.first_hold.mean():.0%}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
