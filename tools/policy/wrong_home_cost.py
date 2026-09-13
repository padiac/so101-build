r"""What did homing to the wrong dataset's start pose cost?

On 2026-09-12 every rollout of the 304-episode models homed with so101_v2,
because the training copy is named so101_color6_304_trim and no local directory
matched. wrist_roll was parked at +18 where the training data starts at -66 and
never once leaves -97..-38.

This asks how much that displaces the plan, using the real start frames of real
episodes. Two states are compared: the one recorded, and the same one with the
joints moved to where homing actually left them. The pictures are held at the
recorded ones, which understates the damage -- the wrist camera is bolted to
wrist_roll, so on the robot the view was rotated too.

The answer only means something against a scale, so it is reported against two:
the model's own sampling noise, and how far apart two different episodes' plans
are. Moving further than one episode is from another means the arm began each
run somewhere its own training never put it.

    .\.venv-win\Scripts\python.exe tools\policy\wrong_home_cost.py [checkpoint ...]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DS = ROOT / "datasets" / "so101_color6"
WRONG_DS = ROOT / "datasets" / "so101_v2"        # what homing actually used
EPISODES = int(os.environ.get("EPISODES", 24))
SAMPLES = int(os.environ.get("SAMPLES", 8))
MODELS = sys.argv[1:] or ["smolvla_color6_304_100k"]


def frame_at(info, ep_row, key, t, ds):
    import av                                    # after torch, or Windows segfaults

    v = ds / info["video_path"].format(
        video_key=key,
        chunk_index=int(ep_row[f"videos/{key}/chunk_index"]),
        file_index=int(ep_row[f"videos/{key}/file_index"]))
    t0 = float(ep_row[f"videos/{key}/from_timestamp"]) + t
    with av.open(str(v)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s)
        for f in c.decode(video=0):
            if float(f.pts * s.time_base) >= t0 - 1e-6:
                a = f.to_ndarray(format="rgb24").astype("float32") / 255.0
                return torch.from_numpy(a).permute(2, 0, 1)
    raise RuntimeError("no frame")


def start_pose(ds):
    d = pd.concat([pd.read_parquet(f) for f in sorted((ds / "data").rglob("*.parquet"))],
                  ignore_index=True)
    first = np.stack([np.asarray(r.sort_values("frame_index")
                                 ["observation.state"].to_numpy()[0], dtype=float)
                      for _, r in d.groupby("episode_index")])
    return np.median(first, axis=0)


info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
names = info["features"]["observation.state"]["names"]
ep = pd.concat([pd.read_parquet(f) for f in
                sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
               ignore_index=True).set_index("episode_index")
data = pd.concat([pd.read_parquet(f) for f in
                  sorted((DS / "data").rglob("*.parquet"))], ignore_index=True)
tasks = pd.read_parquet(DS / "meta" / "tasks.parquet")
task_of = {int(v): k for k, v in tasks["task_index"].items()}
cache = {int(e): r.sort_values("frame_index") for e, r in data.groupby("episode_index")}
chosen = sorted(cache)[::max(1, len(cache) // EPISODES)][:EPISODES]

right = start_pose(DS)
wrong = start_pose(WRONG_DS)
print("start pose, recorded vs where homing left it")
for j, n in enumerate(names):
    flag = "  <-- " + f"{abs(wrong[j] - right[j]):.0f} units" if abs(wrong[j] - right[j]) > 10 else ""
    print(f"  {n:<18}{right[j]:7.1f}  {wrong[j]:7.1f}{flag}")
print(f"\n{len(chosen)} episodes, {SAMPLES} plans each, pictures held at the "
      "recorded ones")

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

IMAGES, TASK, STATE = [], [], []
for e in chosen:
    IMAGES.append({f"observation.images.{cam}": frame_at(info, ep.loc[e], key, 0.0, DS)
                   for cam, key in (("camera1", "observation.images.top"),
                                    ("camera2", "observation.images.wrist"))})
    TASK.append(task_of[int(cache[e]["task_index"].iloc[0])])
    STATE.append(np.asarray(cache[e]["observation.state"].to_numpy()[0], dtype=float))

for name in MODELS:
    d = ROOT / "policies" / name
    policy = SmolVLAPolicy.from_pretrained(str(d)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(d),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})

    def plans(k, state):
        obs = {"observation.state": torch.tensor(state, dtype=torch.float32),
               "task": TASK[k], "robot_type": "so101_follower", **IMAGES[k]}
        out = []
        with torch.inference_mode():
            for _ in range(SAMPLES):
                policy.reset()
                out.append(post(policy.predict_action_chunk(pre(dict(obs))))
                           .squeeze(0).float().cpu().numpy())
        return np.stack(out)

    moved, noise, good = [], [], []
    for k in range(len(chosen)):
        a = plans(k, STATE[k])
        b = plans(k, STATE[k] + (wrong - right))
        moved.append(float(np.abs(a.mean(0) - b.mean(0)).mean()))
        noise.append(float(a.std(0).mean()))
        good.append(a.mean(0))
    between = float(np.mean([np.abs(good[i] - good[j]).mean()
                             for i in range(len(good)) for j in range(i + 1, len(good))]))

    print(f"\n{name}")
    print(f"  the wrong start pose moves the plan by : {np.mean(moved):6.2f} units")
    print(f"  the model's own sampling noise         : {np.mean(noise):6.2f}")
    print(f"  one episode's plan from another's      : {between:6.2f}")
    print(f"  wrong pose / difference between episodes: "
          f"{np.mean(moved) / between if between else 0:5.2f}")
    del policy
    torch.cuda.empty_cache()

print("\nAbove 1.00 the arm starts each run further from its training than two")
print("different demonstrations are from each other.")
