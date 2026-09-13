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
import os
import sys
import time

import numpy as np
import pandas as pd


# The move home runs from wherever the last run left the arm, which is extended
# over the table with the wrist rolled over. Driving every joint at once traces a
# straight line in joint space, and that line goes through the table and through
# the wrist camera -- a camera has been hit and screws shaken loose that way. So
# fold the arm up first, then turn the wrist, and only then swing the base.
STAGES = (
    ("lift",  ("shoulder_lift", "elbow_flex")),
    ("wrist", ("wrist_flex", "wrist_roll")),
    ("home",  None),                       # None: every joint, to the start pose
)


def plan_stages(names, cur, tgt, waypoint=None):
    """Each leg of the move: (label, goal pose, which joints must arrive).

    Joints a leg does not name hold the position the previous leg left them in,
    rather than tracking the live measurement, so gravity droop does not walk
    them downward while they wait.
    """
    wp = np.array(tgt, dtype=float)
    for i, n in enumerate(names):
        if waypoint and f"{n}.pos" in waypoint:
            wp[i] = float(waypoint[f"{n}.pos"])

    plan, hold = [], np.array(cur, dtype=float)
    for label, movers in STAGES:
        if movers is None:
            goal, idx = np.array(tgt, dtype=float), tuple(range(len(names)))
        else:
            idx = tuple(i for i, n in enumerate(names) if n in movers)
            goal = hold.copy()
            for i in idx:
                goal[i] = wp[i]
        plan.append((label, goal, idx))
        hold = goal.copy()
    return plan


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
    ap.add_argument("--save-waypoint", action="store_true",
                    help="record the arm's CURRENT pose as the safe intermediate "
                         "pose for live_task.py and exit without moving")
    a = ap.parse_args()

    tgt, starts = target_pose(a.dataset)

    # Optional intermediate pose for the lift and wrist legs, captured from the
    # arm with --save-waypoint. How high to lift and where to park the wrist
    # depends on what is on the desk, which no dataset records.
    waypoint = None
    wp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "home_waypoint.json")
    if os.path.exists(wp_file) and not a.save_waypoint:
        import json
        with open(wp_file, encoding="utf-8") as f:
            waypoint = json.load(f)
        print("waypoint: %s" % ", ".join("%s %.0f" % (k.split(".")[0], v)
                                         for k, v in waypoint.items()))

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

        if a.save_waypoint:
            # Between two instructions the arm is folded up and the wrist turned
            # before the base swings, so it does not sweep the path that has
            # already cost a wrist camera. Where "up" and "turned" are depends on
            # what is on the desk, so it is captured from the arm: put it where
            # it should pass through, then run this.
            import json
            out = {f"{n}.pos": round(float(cur[i]), 2) for i, n in enumerate(names)}
            wp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "home_waypoint.json")
            with open(wp, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print("saved waypoint to %s" % wp)
            for n, v in out.items():
                print("  %-16s %8.2f" % (n, v))
            print()
            print("live_task.py will now fold to this pose and turn the wrist here")
            print("before swinging home. Delete the file to go back to the default.")
            return 0

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
        speed = a.max_step * a.fps                    # units per second
        t0 = time.perf_counter()
        for label, goal, idx in plan_stages(names, cur, tgt, waypoint):
            sel = list(idx)
            far = np.abs(goal[sel] - cur[sel]).max()
            if far <= a.tol:
                print("[%s] already there" % label)
                continue
            # Each leg gets a deadline from its own distance. One budget for the
            # whole path is either too tight for a long move or too slack to
            # notice a joint that is not moving at all.
            print("[%s] %.0f units, about %.1fs" % (label, far, far / speed))
            deadline = time.perf_counter() + far / speed + 3.0
            while True:
                obs = dev.get_observation()
                cur = np.array([obs[f"{n}.pos"] for n in names], dtype=float)
                err = goal - cur
                if np.abs(err[sel]).max() <= a.tol:
                    break
                if time.perf_counter() > deadline or time.perf_counter() - t0 > a.timeout:
                    stuck = names[sel[int(np.argmax(np.abs(err[sel])))]]
                    print("[!] %s stalled in stage %s: %.2f units off"
                          % (stuck, label, np.abs(err[sel]).max()))
                    print("    Not moving further. The policy would be starting")
                    print("    from a pose no training episode began in.")
                    return 2
                step = np.clip(err, -a.max_step, a.max_step)
                dev.send_action({f"{n}.pos": float((cur + step)[i])
                                 for i, n in enumerate(names)})
                time.sleep(dt)
        obs = dev.get_observation()
        cur = np.array([obs[f"{n}.pos"] for n in names], dtype=float)
        print("[OK] arrived, max residual %.2f" % np.abs(tgt - cur).max())
        return 0
    finally:
        dev.disconnect()


if __name__ == "__main__":
    sys.exit(main())
