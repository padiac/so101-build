"""Why does the arm lift and then come back?

Two different mechanisms produce that, and they need different fixes:

  (a) the policy commands motion far faster than anything demonstrated, so the
      arm overshoots into poses no demo ever visited, and the next plan --
      correctly, for a state it does not recognise -- retreats;
  (b) successive replans genuinely disagree, and the arm is being pulled back
      and forth at the replan period.

Per-step command deltas separate them: if the rollout's deltas are far larger
than the training set's, (a) is in play. The measured trajectory's reversal
period separates them further -- reversals at the replan interval point at (b),
slower swings point at (a).

ASCII output only.
"""

import glob

import numpy as np
import pandas as pd

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30
N_ACTION_STEPS = 20


def load(root):
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))])
    df = df.sort_values(["episode_index", "frame_index"])
    return (np.stack(df["action"].to_numpy()),
            np.stack(df["observation.state"].to_numpy()),
            df["episode_index"].to_numpy())


ra, rs, _ = load("datasets/rollout_probe")
ta, ts, tep = load("datasets/so101_v3")

rd = np.abs(np.diff(ra, axis=0))
td = []
for e in np.unique(tep):
    td.append(np.abs(np.diff(ta[tep == e], axis=0)))
td = np.concatenate(td)

print("=== per-step command delta: rollout vs demonstrations ===")
print("%-15s %8s %8s %8s %8s %8s" % ("joint", "roll p50", "roll p95", "demo p50", "demo p95", "ratio"))
for j, nm in enumerate(JOINTS):
    r50, r95 = np.percentile(rd[:, j], [50, 95])
    d50, d95 = np.percentile(td[:, j], [50, 95])
    print("%-15s %8.2f %8.2f %8.2f %8.2f %8.1fx"
          % (nm, r50, r95, d50, d95, r95 / max(d95, 1e-6)))
print()

print("=== measured trajectory: how often does it turn around? ===")
print("replan interval = %d frames (%.2f s)" % (N_ACTION_STEPS, N_ACTION_STEPS / FPS))
print("%-15s %10s %12s %12s" % ("joint", "range", "reversals", "period (s)"))
for j, nm in enumerate(JOINTS):
    c = rs[:, j]
    # smooth lightly so servo noise is not counted as a turn-around
    k = 9
    sm = np.convolve(c, np.ones(k) / k, mode="valid")
    d = np.diff(sm)
    rev = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
    period = (2 * len(sm) / rev / FPS) if rev else float("inf")
    print("%-15s %10.2f %12d %12.2f" % (nm, c.max() - c.min(), rev, period))
print()
print("period near %.2f s -> replans disagree" % (N_ACTION_STEPS / FPS))
print("period much longer -> it goes too far, then retreats from an unfamiliar pose")
