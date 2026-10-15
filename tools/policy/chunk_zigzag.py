r"""Is the frame-to-frame zigzag in the commands already inside the model's plan?

Rollout s01 commanded 75-90 joint direction reversals per second against 1 per
second in the demonstrations. Two places could put them there: the model's own
predicted chunk, or something between the model and the servos (normalisation,
clamping, the action queue). analyze_rollout.py Q2 asks exactly this but only
loads ACT.

This feeds recorded observations back to a SmolVLA checkpoint offline, takes the
predicted chunk, and counts reversals per second inside it with the same rule the
online count used (arm joints, steps larger than MOVE_THRESH in sum). If the
offline chunk zigzags as much as the online command, the model planned it.

It also measures a checkpoint on its OWN training scene, so a known-good policy
(smolvla_v3only_20k picks on the robot) says whether zigzag is what failure looks
like or just what SmolVLA looks like.

    .\.venv-win\Scripts\python.exe tools\policy\chunk_zigzag.py rollout datasets\rollouts\s01\03_colour_blue policies\smolvla_color6_453_40k
    .\.venv-win\Scripts\python.exe tools\policy\chunk_zigzag.py demo datasets\so101_v3 policies\smolvla_v3only_20k
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FPS = 30
MOVE_THRESH = 0.15
SAMPLES = 8


def flips_per_s(x: np.ndarray) -> float:
    d = np.diff(x[:, :5], axis=0)
    mv = np.abs(d).sum(1) > MOVE_THRESH
    s = np.sign(d[mv])
    return float(np.sum(s[1:] * s[:-1] < 0) / (len(x) / FPS)) if len(s) > 1 else 0.0


def frame(ds: Path, info: dict, ep_row, key: str, t: float):
    import av

    v = ds / info["video_path"].format(video_key=key,
                                      chunk_index=int(ep_row[f"videos/{key}/chunk_index"]),
                                      file_index=int(ep_row[f"videos/{key}/file_index"]))
    target = float(ep_row[f"videos/{key}/from_timestamp"]) + t
    with av.open(str(v)) as c:
        s = c.streams.video[0]
        c.seek(int(max(target - 1.0, 0) / s.time_base), stream=s)
        for f in c.decode(video=0):
            if float(f.pts * s.time_base) >= target - 1e-6:
                a = f.to_ndarray(format="rgb24").astype("float32") / 255.0
                return torch.from_numpy(a).permute(2, 0, 1)
    raise RuntimeError("no frame")


def main(argv) -> int:
    mode, ds, ckpt = argv[0], Path(argv[1]), Path(argv[2])
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    meta = pd.concat([pd.read_parquet(f) for f in sorted((ds / "meta" / "episodes").rglob("*.parquet"))],
                     ignore_index=True).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in sorted((ds / "data").rglob("*.parquet"))], ignore_index=True)

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to("cuda").eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})

    eps = sorted(meta.index)
    picks = [(eps[0], None)] if mode == "rollout" else [(e, None) for e in eps[:: max(1, len(eps) // 4)][:4]]
    offline, online = [], []
    for e, _ in picks:
        g = data[data.episode_index == e].sort_values("frame_index")
        st = np.stack(g["observation.state"].to_numpy())
        ac = np.stack(g["action"].to_numpy())
        task = meta.loc[e]["tasks"][0]
        starts = list(range(30, len(st) - 60, max(1, (len(st) - 90) // 6)))[:6]
        for f0 in starts:
            obs = {"observation.state": torch.tensor(st[f0], dtype=torch.float32),
                   "task": task, "robot_type": "so101_follower",
                   "observation.images.camera1": frame(ds, info, meta.loc[e], "observation.images.top", f0 / FPS),
                   "observation.images.camera2": frame(ds, info, meta.loc[e], "observation.images.wrist", f0 / FPS)}
            with torch.inference_mode():
                for _ in range(SAMPLES):
                    policy.reset()
                    chunk = post(policy.predict_action_chunk(pre(dict(obs)))).squeeze(0).float().cpu().numpy()
                    offline.append(flips_per_s(chunk))
            online.append(flips_per_s(ac[f0:f0 + chunk.shape[0]]))
    what = "online command" if mode == "rollout" else "demonstrated action"
    print(f"{ckpt.name} on {ds.name} ({mode})")
    print(f"  offline predicted chunk : {np.median(offline):6.1f} reversals/s  "
          f"(IQR {np.percentile(offline, 25):.1f}-{np.percentile(offline, 75):.1f}, {len(offline)} chunks)")
    print(f"  {what:<23}: {np.median(online):6.1f} reversals/s  (same windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
