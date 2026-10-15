r"""How rough is each policy's plan, across policy classes, on the same frames?

On the robot the diffusion model is visibly the smoothest of the four trained on
so101_v3, SmolVLA at 40k is acceptable where 20k was not, and ACT and SmolVLA are
hard to tell apart. That ordering was felt, not measured. This measures it
offline, from the same real frames, so the next comparison does not need the arm.

Two numbers per checkpoint, both properties of one predicted chunk:

  reversals/s  how often the commanded direction flips, summed over the arm
               joints -- the frame-to-frame sawtooth (tools/policy/chunk_zigzag.py)
  roughness    mean absolute second difference of the commanded pose -- how much
               the path bends inside one plan (tools/policy/tremor.py)
  own future   how often the plan is closer to THIS moment's recorded future than
               to the other sampled moments' futures. Smoothness says nothing
               about aim, and on the robot a policy can be smooth and still reach
               the wrong place; 50% is chance.
  ms per plan  wall time for one forward pass. A policy that needs half a second
               to replan leaves the arm running on a stale plan, which is what
               "very slow" looks like from outside.

The demonstrations are scored the same way as the floor. Camera keys differ by
policy class, so they are read from each checkpoint rather than assumed.

    .\.venv-win\Scripts\python.exe tools\policy\smoothness_across_policies.py [ckpt ...]
"""

import json
import os
import sys
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Which recordings supply the frames. Colour checkpoints must be scored on
# the colour scenes, not on so101_v3.
DS = ROOT / "datasets" / os.environ.get("DATASET", "so101_v3")
FPS = 30
MOVE_THRESH = 0.15
SAMPLES = int(os.environ.get("SAMPLES", 6))
EPISODES = int(os.environ.get("EPISODES", 4))
STARTS = 6
HORIZON = 30                    # scored over the shortest plan any of these produce
DEFAULT = ["act_v3rand_100k", "smolvla_v3only_20k", "smolvla_v3only_40k",
           "diffusion_v3_100k"]


def flips_per_s(x: np.ndarray) -> float:
    d = np.diff(x[:, :5], axis=0)
    mv = np.abs(d).sum(1) > MOVE_THRESH
    s = np.sign(d[mv])
    return float(np.sum(s[1:] * s[:-1] < 0) / (len(x) / FPS)) if len(s) > 1 else 0.0


def roughness(x: np.ndarray) -> float:
    return float(np.abs(np.diff(np.diff(x, axis=0), axis=0)).mean())


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


info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
meta = pd.concat([pd.read_parquet(f) for f in sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
                 ignore_index=True).set_index("episode_index")
data = pd.concat([pd.read_parquet(f) for f in sorted((DS / "data").rglob("*.parquet"))],
                 ignore_index=True)
eps = sorted({int(e) for e in data.episode_index})[:: max(1, len(set(data.episode_index)) // EPISODES)][:EPISODES]

points = []                                      # (episode, frame index)
for e in eps:
    g = data[data.episode_index == e].sort_values("frame_index")
    n = len(g)
    points += [(e, f0) for f0 in range(30, n - 70, max(1, (n - 100) // STARTS))[:STARTS]]

demo_f, demo_r, FUT = [], [], []
for e, f0 in points:
    g = data[data.episode_index == e].sort_values("frame_index")
    a = np.stack(g["action"].to_numpy())[f0:f0 + 50]
    demo_f.append(flips_per_s(a))
    demo_r.append(roughness(a))
    FUT.append(a[:HORIZON])
FUT = np.stack(FUT)


def own_future(k, plan):
    d = np.sqrt(((FUT - plan[None, :HORIZON]) ** 2).mean(axis=(1, 2)))
    return float((np.delete(d, k) > d[k]).mean())

print(f"{len(points)} moments from {len(eps)} episodes, {SAMPLES} plans each\n")
print(f"  {'checkpoint':<24}{'reversals/s':>13}{'roughness':>12}{'own future':>12}{'ms/plan':>10}")
print(f"  {'demonstration':<24}{np.median(demo_f):13.1f}{np.median(demo_r):12.3f}"
      f"{'100%':>12}{'-':>10}")

from lerobot.configs.policies import PreTrainedConfig                  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

for name in (sys.argv[1:] or DEFAULT):
    # A published name under policies/, or a path to any checkpoint directory:
    # every run here keeps its intermediate checkpoints, so a curve over training
    # costs nothing but inference.
    d = ROOT / "policies" / name
    if not (d / "config.json").exists():
        d = pathlib.Path(name)
    if not (d / "config.json").exists():
        raise SystemExit(f"{name}: no checkpoint here")
    cfg = PreTrainedConfig.from_pretrained(str(d))
    policy = get_policy_class(cfg.type).from_pretrained(str(d)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(d),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    # camera1/camera2 for SmolVLA, top/wrist for ACT and diffusion: ask the
    # checkpoint instead of assuming, and pair them with the recorded streams in
    # the order they were recorded, which is what the rename map did at training.
    want = sorted(k for k in cfg.input_features if ".images." in k)
    have = ["observation.images.top", "observation.images.wrist"]
    pairs = list(zip(want, have))

    # Diffusion conditions on n_obs_steps frames, not one, and asserts on the
    # count rather than padding. Feeding it the preceding frames through
    # select_action fills its observation queue the way a rollout would.
    nobs = int(getattr(cfg, "n_obs_steps", 1) or 1)

    def observation(e, f):
        g = data[data.episode_index == e].sort_values("frame_index")
        obs = {"observation.state": torch.tensor(
                   np.stack(g["observation.state"].to_numpy())[f], dtype=torch.float32),
               "task": meta.loc[e]["tasks"][0], "robot_type": "so101_follower"}
        for key, src in pairs:
            obs[key] = frame(info, meta.loc[e], src, f / FPS)
        return obs

    def plan_single(obs):
        return post(policy.predict_action_chunk(pre(dict(obs)))).squeeze(0)

    def plan_with_history(history):
        """The actions a multi-observation policy would actually send.

        predict_action_chunk cannot be called directly here: with its queues
        filled it rebuilds the batch from the keys the caller passed, and the
        stacked-image key it needs internally is not one of them. Driving it
        through select_action, the way a rollout does, and taking the action it
        returned plus what is still queued gives the executed part of the plan.
        """
        for past in history[:-1]:
            policy.select_action(pre(dict(past)))
        first = policy.select_action(pre(dict(history[-1])))
        rest = list(policy._queues["action"])
        return post(torch.stack([first] + rest, dim=1)).squeeze(0)

    import time
    fl, ro, rk, ms = [], [], [], []
    for k, (e, f0) in enumerate(points):
        history = [observation(e, f) for f in range(f0 - nobs + 1, f0 + 1)]
        with torch.inference_mode():
            for _ in range(SAMPLES):
                policy.reset()
                t0 = time.perf_counter()
                chunk = (plan_with_history(history) if nobs > 1
                         else plan_single(history[-1]))
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t0) * 1000)
                chunk = chunk.float().cpu().numpy()
                fl.append(flips_per_s(chunk))
                ro.append(roughness(chunk))
                rk.append(own_future(k, chunk))
    label = name if len(name) < 24 else "..." + name[-21:]
    print(f"  {label:<24}{np.median(fl):13.1f}{np.median(ro):12.3f}"
          f"{np.mean(rk):11.0%}{np.median(ms):10.0f}")
    del policy
    torch.cuda.empty_cache()

print("\nBoth are medians over plans. The demonstration row is the floor: that is")
print("how much a human-driven arm bends and reverses over the same 50 steps.")
