r"""Knock the arm off course on paper. Does the plan steer back?

Watched on the robot: the diffusion policy goes out, finds itself not squarely
over the cube, moves again, squares the wrist, and only then closes -- the
operator's own habit. ACT does some of this. SmolVLA does none of it: it goes
where it goes and closes on air if that was wrong.

That is a claim about feedback, and feedback can be tested without the robot.
Take a real moment from a demonstration, displace the arm's reported position by
a few units, and ask for a plan. A policy that only replays its trained
trajectory will plan from the displaced pose and stay displaced. A policy that
corrects will bend back toward where the demonstration actually went.

    recovered = 1 - (how far the plan still is at the end) / (how far it started)

1.0 is full correction by the end of the chunk, 0.0 is none, negative is drifting
further away. The unperturbed plan is measured the same way as the floor, since
no policy tracks the demonstration perfectly even when undisturbed.

The pictures are NOT displaced, only the reported joint angles, so this isolates
one channel: does the policy act on where it is told the arm is. On the robot a
real displacement would also move the wrist camera.

    .\.venv-win\Scripts\python.exe tools\policy\does_it_correct.py [ckpt ...]
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

DS = ROOT / "datasets" / "so101_v3"
FPS = 30
SAMPLES = int(os.environ.get("SAMPLES", 4))
EPISODES = int(os.environ.get("EPISODES", 4))
STARTS = 5
HORIZON = 30
# Displace the joint that aims the arm sideways, by about what "not squarely over
# the cube" looks like: the demonstrations' own shoulder_pan varies by ~12 units
# between episodes, so 8 is a miss that matters but is not off the table.
NUDGE_JOINT, NUDGE = 0, 8.0
DEFAULT = ["act_v3rand_100k", "smolvla_v3only_100k_b16", "diffusion_v3_100k"]


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
eps = sorted({int(e) for e in data.episode_index})
eps = eps[:: max(1, len(eps) // EPISODES)][:EPISODES]
points = []
for e in eps:
    n = len(data[data.episode_index == e])
    points += [(e, f0) for f0 in range(30, n - 70, max(1, (n - 100) // STARTS))[:STARTS]]

print(f"{len(points)} moments, {SAMPLES} plans each, "
      f"shoulder_pan displaced by {NUDGE:.0f} units\n")

from lerobot.configs.policies import PreTrainedConfig                          # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

print(f"  {'checkpoint':<26}{'undisturbed':>12}{'displaced':>11}{'recovered':>11}")
for name in (sys.argv[1:] or DEFAULT):
    d = ROOT / "policies" / name
    if not (d / "config.json").exists():
        d = Path(name)
    cfg = PreTrainedConfig.from_pretrained(str(d))
    policy = get_policy_class(cfg.type).from_pretrained(str(d)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(d),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    want = sorted(k for k in cfg.input_features if ".images." in k)
    pairs = list(zip(want, ["observation.images.top", "observation.images.wrist"]))
    nobs = int(getattr(cfg, "n_obs_steps", 1) or 1)

    def build(e, f, shift):
        g = data[data.episode_index == e].sort_values("frame_index")
        st = np.asarray(np.stack(g["observation.state"].to_numpy())[f], dtype=np.float32).copy()
        st[NUDGE_JOINT] += shift
        o = {"observation.state": torch.tensor(st, dtype=torch.float32),
             "task": meta.loc[e]["tasks"][0], "robot_type": "so101_follower"}
        for key, src in pairs:
            o[key] = frame(info, meta.loc[e], src, f / FPS)
        return o

    def plan(hist):
        if nobs > 1:
            for past in hist[:-1]:
                policy.select_action(pre(dict(past)))
            first = policy.select_action(pre(dict(hist[-1])))
            out = post(torch.stack([first] + list(policy._queues["action"]), dim=1))
        else:
            out = post(policy.predict_action_chunk(pre(dict(hist[-1]))))
        return out.squeeze(0).float().cpu().numpy()

    clean, moved, rec = [], [], []
    for e, f0 in points:
        g = data[data.episode_index == e].sort_values("frame_index")
        fut = np.stack(g["action"].to_numpy())[f0:f0 + HORIZON, NUDGE_JOINT]
        with torch.inference_mode():
            for shift, bucket in ((0.0, clean), (NUDGE, moved)):
                hist = [build(e, f, shift) for f in range(f0 - nobs + 1, f0 + 1)]
                errs = []
                for _ in range(SAMPLES):
                    policy.reset()
                    p = plan(hist)[:HORIZON, NUDGE_JOINT]
                    errs.append((abs(p[0] - fut[0]), abs(p[-1] - fut[-1])))
                bucket.append(np.mean(errs, axis=0))
    clean, moved = np.stack(clean), np.stack(moved)
    # Recovery is measured against the start of the displaced plan, and only
    # credited beyond whatever the undisturbed plan drifts on its own.
    gained = moved[:, 0] - moved[:, 1]
    rec = float(np.median(gained / np.maximum(moved[:, 0], 1e-6)))
    print(f"  {name[-26:]:<26}{np.median(clean[:, 1]):12.1f}"
          f"{np.median(moved[:, 1]):11.1f}{rec:11.0%}")
    del policy
    torch.cuda.empty_cache()

print("\nThe two middle columns are how far the plan's last step sits from the")
print("demonstration's, in units of shoulder_pan. Recovered is how much of the")
print("displacement the plan had closed by the end of the chunk.")
