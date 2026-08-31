#!/usr/bin/env python3
"""Move the follower to the pose every training episode started from.

WHY THIS EXISTS
    Measured over the 30 recorded episodes, the start pose is extremely tight:

        shoulder_lift  min -100.00  max -99.65   spread 0.35
        elbow_flex       85.93        94.02      spread 8.09

    Every episode began from essentially one configuration -- wherever the
    leader arm rests between takes. The policy has therefore never seen a
    state that does not start there.

    At eval the follower starts wherever it was left. If that is outside this
    distribution, the very first observation is out of distribution, and ACT
    then executes 100 open-loop steps (3.3 s) planned from it. The visible
    symptom is exactly what you would expect: the arm drifts up and down and
    never approaches the object.

    So: park the arm at the training start pose before handing control to the
    policy. This is not a workaround -- matching the start distribution is a
    real precondition of behaviour cloning.

The target is recomputed from the dataset each run, so it cannot drift out of
sync with the data the way a hardcoded constant would.

ASCII output only (the Windows console is GBK).

Usage:
    python home.py --port COM7
    python home.py --port COM7 --dataset datasets/so101_pickplace
"""

import argparse
import glob
import sys
import time

import numpy as np
import pandas as pd


def target_pose(root: str):
    """Median start pose across all recorded episodes, per joint."""
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {root}/data")
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df.sort_values(["episode_index", "frame_index"])
    state = np.stack(df["observation.state"].to_numpy())
    ep = df["episode_index"].to_numpy()
    starts = np.stack([state[ep == e][0] for e in np.unique(ep)])
    # median, not mean: robust to a single episode that began oddly
    return np.median(starts, axis=0), starts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--dataset", default="datasets/so101_pickplace")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-step", type=float, default=1.5,
                    help="max units per tick per joint; keeps the move gentle")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--tol", type=float, default=3.5,
                    help="per-joint arrival tolerance (friction and gravity droop leave ~2 units of residual, so 2.0 was too tight)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without moving")
    a = ap.parse_args()

    tgt, starts = target_pose(a.dataset)

    # Torque writes drop packets during the power-on surge; lerobot_patch adds
    # retries. Every script that opens the bus needs it (README #1).
    import lerobot_patch
    lerobot_patch.apply(verbose=False)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    dev = SO101Follower(SO101FollowerConfig(
        port=a.port, id="my_follower", max_relative_target=None))
    dev.connect(calibrate=False)

    try:
        obs = dev.get_observation()
        names = [k[:-4] for k in obs if k.endswith(".pos")]
        # keep the dataset's joint order, which is the bus order
        order = list(dev.bus.motors.keys())
        names = [n for n in order if n in names]
        if len(names) != len(tgt):
            print("[X] joint count mismatch: robot %d, dataset %d" % (len(names), len(tgt)))
            return 1

        cur = np.array([obs[f"{n}.pos"] for n in names], dtype=float)
        print("%-15s %9s %9s %9s" % ("joint", "current", "target", "delta"))
        for i, n in enumerate(names):
            print("%-15s %9.2f %9.2f %9.2f" % (n, cur[i], tgt[i], tgt[i] - cur[i]))
        print()
        print("max delta %.2f   moving at <= %.1f units/tick @ %.0f fps"
              % (np.abs(tgt - cur).max(), a.max_step, a.fps))
        print()
        if a.dry_run:
            print("[dry-run] not moving.")
            return 0

        dt = 1.0 / a.fps
        t0 = time.perf_counter()
        while True:
            obs = dev.get_observation()
            cur = np.array([obs[f"{n}.pos"] for n in names], dtype=float)
            err = tgt - cur
            if np.abs(err).max() <= a.tol:
                print("[OK] arrived, max residual %.2f" % np.abs(err).max())
                return 0
            if time.perf_counter() - t0 > a.timeout:
                print("[!] timeout after %.0fs, max residual %.2f"
                      % (a.timeout, np.abs(err).max()))
                print("    Not fatal, but the policy will start slightly off-distribution.")
                return 2
            step = np.clip(err, -a.max_step, a.max_step)
            goal = cur + step
            dev.send_action({f"{n}.pos": float(goal[i]) for i, n in enumerate(names)})
            time.sleep(dt)
    finally:
        dev.disconnect()


if __name__ == "__main__":
    sys.exit(main())
