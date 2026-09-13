r"""Can the instruction be guessed from where the arm went?

If each colour tends to sit in its own part of the workspace, a policy can solve
the task from geometry alone and never read the language -- the data looks fine
and trains fine, and the language input stays dead. This is the one design error
that silently wastes a whole recording session, so it is worth five minutes.

The test: take the arm pose at the moment of the grasp, and ask how well a simple
classifier separates the instructions. Chance is 1/n_tasks. Much above chance
means colour and position are entangled.

    .\.venv-win\Scripts\python.exe tools\dataset\instruction_leak.py datasets\so101_color6
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def grasp_poses(root: Path):
    """Arm pose at the first full gripper close after the widest opening."""
    d = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))],
                  ignore_index=True)
    A = np.stack(d["action"].to_numpy())
    S = np.stack(d["observation.state"].to_numpy())
    df = pd.DataFrame({"ep": d.episode_index.values, "fi": d.frame_index.values,
                       "task": d.task_index.values, "ag": A[:, 5]})
    for k, n in enumerate(J):
        df["s_" + n] = S[:, k]

    rows = []
    for ep, g in df.groupby("ep"):
        g = g.sort_values("fi")
        ag = g.ag.to_numpy()
        pk = int(np.argmax(ag))
        after = np.where(ag[pk:] < 5.0)[0]
        if not len(after):
            continue
        r = g.iloc[pk + int(after[0])]
        rows.append(dict(ep=ep, task=int(r.task), **{n: float(r["s_" + n]) for n in J}))
    return pd.DataFrame(rows)


def main(root: Path) -> int:
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    names = {int(v): k for k, v in zip(tasks.index, tasks.task_index)}
    G = grasp_poses(root)
    n_task = G.task.nunique()
    chance = 1.0 / n_task
    print(f"grasp pose recovered for {len(G)} episodes across {n_task} instructions")
    print(f"chance level = {chance:.1%}\n")

    print("shoulder_pan at grasp -- the axis that says WHERE the cube was")
    print(f"{'instruction':<52} {'n':>3} {'min':>7} {'p25':>7} {'median':>7} {'p75':>7} {'max':>7}")
    for t, g in G.groupby("task"):
        v = g.shoulder_pan
        short = names.get(t, str(t))[:50]
        print(f"{short:<52} {len(g):>3} {v.min():7.1f} {v.quantile(.25):7.1f} "
              f"{v.median():7.1f} {v.quantile(.75):7.1f} {v.max():7.1f}")

    # Leave-one-out nearest neighbour on the grasp pose. Deliberately simple: if
    # even this separates the instructions, a 450M model certainly will.
    X = G[J].to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = G.task.to_numpy()
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    pred = y[d2.argmin(1)]
    acc = (pred == y).mean()

    print()
    print(f"1-NN on grasp pose, leave-one-out:  {acc:.1%}   (chance {chance:.1%})")
    ratio = acc / chance
    if ratio < 1.5:
        print("  -> position carries little about the instruction. The language input")
        print("     is doing real work. This is what you want.")
    elif ratio < 2.5:
        print("  -> mild entanglement. Usable, but some colours sit in preferred spots.")
    else:
        print("  -> STRONG entanglement: the instruction is largely predictable from")
        print("     geometry alone, so the policy can ignore the language and still")
        print("     score well. Re-check how the cubes were placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/so101_color6")))
