r"""From the resting start pose, does the policy plan to move at all?

On the robot, diffusion_emptybowl70_100k sat completely still. It was trained
on recordings that were NOT trimmed, so every episode opens with ~4.6 s of the
operator getting ready while the arm rests at home -- frames whose label is
"keep waiting". At deployment the arm rests at home too, sees exactly those
frames, and if it plans "wait" it never leaves, because waiting does not change
what it sees. That loop does not depend on averaging: a policy that samples a
mode can lock into the "wait" mode just as surely.

So feed every checkpoint the same input -- the first frames of the untrimmed
recordings, which is what the arm sees when it starts -- and measure how far the
plan travels. The trimmed model, which did move on the robot, is the control.

    .\.venv-win\Scripts\python.exe tools\policy\does_it_start.py [ckpt ...]
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

DS = ROOT / "datasets" / os.environ.get("DATASET", "so101_emptybowl")
FPS = 30
EPISODES = int(os.environ.get("EPISODES", 12))
SAMPLES = int(os.environ.get("SAMPLES", 4))
# The two untrimmed models that sat still on 2026-10-14 are archived under
# policies_archive/; pass their paths explicitly to re-measure them.
DEFAULT = ["diffusion_emptybowl_100k", "diffusion_emptybowl70_100k_trim"]


def frame(info, ep_row, key, t):
    import av                                    # after torch, or Windows segfaults

    v = DS / info["video_path"].format(
        video_key=key,
        chunk_index=int(ep_row[f"videos/{key}/chunk_index"]),
        file_index=int(ep_row[f"videos/{key}/file_index"]))
    target = float(ep_row[f"videos/{key}/from_timestamp"]) + t
    with av.open(str(v)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(max(target - 1.0, 0) / s.time_base), stream=s)
        for f in c.decode(video=0):
            if float(f.pts * s.time_base) >= target - 1e-6:
                a = f.to_ndarray(format="rgb24").astype("float32") / 255.0
                return torch.from_numpy(a).permute(2, 0, 1)
    raise RuntimeError("no frame")


def travel(chunk):
    """How far the plan takes the arm from where it starts, for the joint that
    goes furthest. Gripper excluded.

    This used to be path length -- the sum of every step's change. That counts
    jitter: a plan that trembles in place by a unit or two per step adds up to
    20-30 over 50 steps, and on 2026-10-16 a model that sat still on the robot
    scored 29 and "moves 100%". Excursion from the start does not accumulate
    trembling: it is the distance the arm would actually travel."""
    arm = chunk[:, :5]
    return float(np.abs(arm - arm[0]).max())


info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
meta = pd.concat([pd.read_parquet(f) for f in sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
                 ignore_index=True).set_index("episode_index")
data = pd.concat([pd.read_parquet(f) for f in sorted((DS / "data").rglob("*.parquet"))],
                 ignore_index=True)
eps = sorted({int(e) for e in data.episode_index} - {11})
eps = eps[:: max(1, len(eps) // EPISODES)][:EPISODES]

# What the demonstrations themselves do over the same window from frame 0: the
# operator is still getting ready, so this is small, and that is the point.
demo = []
for e in eps:
    a = np.stack(data[data.episode_index == e].sort_values("frame_index")["action"].to_numpy())
    demo.append(travel(a[:50]))

from lerobot.configs.policies import PreTrainedConfig                          # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

print(f"{len(eps)} episodes, first frames of the untrimmed recording, {SAMPLES} plans each\n")
print(f"  {'':<28}{'plan travel':>12}{'moves (>5)':>12}")
print(f"  {'demonstration, frames 0-49':<28}{np.median(demo):12.1f}{np.mean(np.array(demo) > 5):12.0%}")

for name in (sys.argv[1:] or DEFAULT):
    d = ROOT / "policies" / name
    cfg = PreTrainedConfig.from_pretrained(str(d))
    policy = get_policy_class(cfg.type).from_pretrained(str(d)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(d),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    want = sorted(k for k in cfg.input_features if ".images." in k)
    pairs = list(zip(want, ["observation.images.top", "observation.images.wrist"]))
    nobs = int(getattr(cfg, "n_obs_steps", 1) or 1)

    def obs_at(e, f):
        g = data[data.episode_index == e].sort_values("frame_index")
        o = {"observation.state": torch.tensor(
                 np.stack(g["observation.state"].to_numpy())[f], dtype=torch.float32),
             "task": "Take all the cubes out of the bowl and put them on the table.",
             "robot_type": "so101_follower"}
        for key, src in pairs:
            o[key] = frame(info, meta.loc[e], src, f / FPS)
        return o

    t = []
    for e in eps:
        hist = [obs_at(e, f) for f in range(nobs)]       # the arm at rest
        with torch.inference_mode():
            for _ in range(SAMPLES):
                policy.reset()
                if nobs > 1:
                    for past in hist[:-1]:
                        policy.select_action(pre(dict(past)))
                    first = policy.select_action(pre(dict(hist[-1])))
                    chunk = post(torch.stack([first] + list(policy._queues["action"]), dim=1))
                else:
                    chunk = post(policy.predict_action_chunk(pre(dict(hist[-1]))))
                t.append(travel(chunk.squeeze(0).float().cpu().numpy()))
    t = np.array(t)
    print(f"  {name:<28}{np.median(t):12.1f}{np.mean(t > 5):12.0%}")
    del policy
    torch.cuda.empty_cache()

print("\nTravel is how far the commanded pose moves over one plan, in joint units.")
print("A policy that will not leave the start pose plans close to nothing here.")
