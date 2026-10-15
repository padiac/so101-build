r"""Does SmolVLA's roughness come from taking only ten flow-matching steps?

SmolVLA integrates its flow in num_steps=10, while the diffusion policy denoises
in 100. Ten steps is a coarse integration of a continuous path, and a coarse
integration of a smooth path is a jagged one -- so part of the gap in measured
roughness could be an inference setting rather than anything learned.

That is cheap to rule in or out: the same checkpoint, the same frames, only
num_steps changed. Nothing is retrained.

    .\.venv-win\Scripts\python.exe tools\policy\flow_steps_sweep.py [checkpoint]
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
MOVE_THRESH = 0.15
SAMPLES = int(os.environ.get("SAMPLES", 6))
EPISODES = int(os.environ.get("EPISODES", 4))
STARTS = 6
STEPS = [int(x) for x in os.environ.get("NUM_STEPS", "10,20,50,100").split(",")]
CKPT = sys.argv[1] if len(sys.argv) > 1 else "smolvla_v3only_40k"


def flips_per_s(x):
    d = np.diff(x[:, :5], axis=0)
    mv = np.abs(d).sum(1) > MOVE_THRESH
    s = np.sign(d[mv])
    return float(np.sum(s[1:] * s[:-1] < 0) / (len(x) / FPS)) if len(s) > 1 else 0.0


def roughness(x):
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
eps = sorted({int(e) for e in data.episode_index})
eps = eps[:: max(1, len(eps) // EPISODES)][:EPISODES]
points = []
for e in eps:
    n = len(data[data.episode_index == e])
    points += [(e, f0) for f0 in range(30, n - 70, max(1, (n - 100) // STARTS))[:STARTS]]

from lerobot.configs.policies import PreTrainedConfig                          # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

d = ROOT / "policies" / CKPT
cfg = PreTrainedConfig.from_pretrained(str(d))
policy = get_policy_class(cfg.type).from_pretrained(str(d)).to("cuda").eval()
pre, post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=str(d),
    preprocessor_overrides={"device_processor": {"device": "cuda"}})
want = sorted(k for k in cfg.input_features if ".images." in k)
pairs = list(zip(want, ["observation.images.top", "observation.images.wrist"]))

obs_cache = []
for e, f0 in points:
    g = data[data.episode_index == e].sort_values("frame_index")
    o = {"observation.state": torch.tensor(
             np.stack(g["observation.state"].to_numpy())[f0], dtype=torch.float32),
         "task": meta.loc[e]["tasks"][0], "robot_type": "so101_follower"}
    for key, src in pairs:
        o[key] = frame(info, meta.loc[e], src, f0 / FPS)
    obs_cache.append(o)

print(f"{CKPT}: {len(points)} moments, {SAMPLES} plans each")
print(f"  {'flow steps':>11}{'reversals/s':>14}{'roughness':>12}{'ms per plan':>13}")
import time

for ns in STEPS:
    policy.config.num_steps = ns
    fl, ro, dt = [], [], []
    with torch.inference_mode():
        for o in obs_cache:
            for _ in range(SAMPLES):
                policy.reset()
                t0 = time.perf_counter()
                chunk = post(policy.predict_action_chunk(pre(dict(o)))).squeeze(0)
                torch.cuda.synchronize()
                dt.append((time.perf_counter() - t0) * 1000)
                chunk = chunk.float().cpu().numpy()
                fl.append(flips_per_s(chunk))
                ro.append(roughness(chunk))
    print(f"  {ns:>11}{np.median(fl):14.1f}{np.median(ro):12.3f}{np.median(dt):13.0f}")

print("\nIf roughness barely moves, ten steps was not the problem and the jaggedness")
print("is in what the model learned to predict.")
