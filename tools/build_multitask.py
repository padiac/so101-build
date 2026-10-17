"""Merge several trimmed datasets into one, each frame tagged with a task vector.

lerobot's diffusion policy takes no instruction. Mixed blindly, recordings
that start from the same scene but want different behaviour teach it
contradictions: so101_color6 takes ONE cube out of a full bowl and goes home,
so101_emptybowl keeps going until the bowl is empty, so101_v3 puts cubes IN.
The policy does accept an extra conditioning vector, observation.environment_state
(the input Push-T uses for keypoints), and appends it to its global condition.
Putting a one-hot task id there lets the shared skill -- reaching, grasping a
cube in a bowl, reading the wrist camera -- be learned from all of them, while
each task keeps its own rules. At deployment the vector is fixed to the task
wanted (see lerobot_patch.apply_task_vector and robot.ps1).

Runs in WSL, on the trimmed copies the training scripts already made:

    .venv/bin/python /mnt/e/Repo/so101-build/tools/build_multitask.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import add_features, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA = Path("/home/padiac/lerobot-train/data")
# Order is the order of the one-hot vector; never reorder once a model exists.
TASKS = [
    ("emptybowl", "so101_emptybowl_trim"),   # take every cube out
    ("take_one", "so101_color6_453_trim"),   # take one (named) cube out
    ("put_in", "so101_v3_trim"),             # put the yellow block in
]
OUT = "so101_multitask_trim"
KEY = "observation.environment_state"


def main() -> int:
    tagged = []
    for i, (name, src) in enumerate(TASKS):
        ds = LeRobotDataset(f"local/{src}", root=DATA / src)
        vec = np.zeros(len(TASKS), dtype=np.float32)
        vec[i] = 1.0
        n = ds.meta.total_frames
        dst = DATA / f"{src}_tagged"
        shutil.rmtree(dst, ignore_errors=True)
        info = {"dtype": "float32", "shape": [len(TASKS)], "names": [t for t, _ in TASKS]}
        out = add_features(ds, {KEY: (np.tile(vec, (n, 1)), info)},
                           output_dir=dst, repo_id=f"local/{src}_tagged")
        print(f"{name:10} {src:24} {out.meta.total_episodes:4} episodes  {n:7} frames  -> {vec.tolist()}")
        tagged.append(out)

    shutil.rmtree(DATA / OUT, ignore_errors=True)
    merged = merge_datasets(tagged, output_repo_id=f"local/{OUT}", output_dir=DATA / OUT)
    want_eps = sum(t.meta.total_episodes for t in tagged)
    want_frames = sum(t.meta.total_frames for t in tagged)
    print(f"merged: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames"
          f" (expected {want_eps}, {want_frames})")
    if merged.meta.total_episodes != want_eps or merged.meta.total_frames != want_frames:
        print("!!! merged counts do not add up")
        return 1
    if KEY not in merged.meta.features:
        print(f"!!! {KEY} missing after merge")
        return 1
    (DATA / OUT / "task_vector.json").write_text(json.dumps({"key": KEY, "names": [t for t, _ in TASKS]}))
    for t in tagged:
        shutil.rmtree(t.root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
