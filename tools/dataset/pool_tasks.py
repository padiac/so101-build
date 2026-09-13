r"""Collapse a dataset's instructions into one, keeping every episode.

so101_color6 is 150 demonstrations of the same motion -- reach into the bowl,
take a cube, put it on the table -- split five ways by colour, so each
instruction has only 30. The model trained on it does not reach for any cube at
all, which is a failure of the motion, not of colour discrimination.

Pooling asks the question that separates those: with all 150 demonstrations of
the motion behind a single instruction, can the motion be learned? Any cube
counts as success.

It makes the task genuinely ambiguous -- one image, five correct answers -- and
that is deliberate. A regression policy would average those and reach the middle
of the bowl, which is exactly the observed symptom; a flow-matching policy like
SmolVLA can represent several modes and should pick one. Either outcome is worth
knowing. And if the policy is not using the language anyway, the task was already
this ambiguous and pooling only makes it visible.

    python tools/dataset/pool_tasks.py --src datasets/so101_color6 \
        --dst datasets/so101_c6pool --task "Pick a cube out of the bowl and put it on the table."
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--task", required=True, help="the single instruction to use")
    ap.add_argument("--force", action="store_true", help="overwrite an existing dst")
    a = ap.parse_args()

    if a.dst.exists():
        if not a.force:
            print(f"{a.dst} exists; pass --force to replace it")
            return 1
        shutil.rmtree(a.dst)

    print(f"copying {a.src} -> {a.dst}")
    # Videos are the bulk and are untouched; only the instruction changes.
    shutil.copytree(a.src, a.dst)

    tasks_file = a.dst / "meta" / "tasks.parquet"
    before = pd.read_parquet(tasks_file)
    print(f"instructions before: {len(before)}")
    for name in before.index:
        print(f"  {name}")

    pd.DataFrame({"task_index": [0]},
                 index=pd.Index([a.task], name="task")).to_parquet(tasks_file)

    n = 0
    for f in sorted((a.dst / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        df["task_index"] = 0
        df.to_parquet(f, index=False)
        n += len(df)
    print(f"\ninstruction after : {a.task}")
    print(f"rewrote {n} frames across the data files")

    # Read it back the way the trainer will, rather than trusting the write.
    after = pd.read_parquet(tasks_file)
    assert len(after) == 1 and after.index[0] == a.task, after
    seen = set()
    eps = set()
    for f in sorted((a.dst / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        seen |= set(df.task_index.unique())
        eps |= set(df.episode_index.unique())
    assert seen == {0}, f"task_index still varies: {sorted(seen)}"
    print(f"verified: one instruction, task_index {sorted(seen)}, "
          f"{len(eps)} episodes kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
