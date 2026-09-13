r"""When the job is done, does the policy plan to stop?

Observed on the arm: it finishes the task and then keeps stirring. The obvious
suspect is the data ending mid-motion, and that is not it -- every so101_v3
episode ends with the arm still for the last half second, and 94-98% of the
others do.

So ask the policy instead. At the final frame of a real episode, where the
recording shows the arm parked, how far does the planned chunk travel? A
mid-episode frame is measured the same way in the same run, as the control that
says the number means anything: there the policy should plan a lot of motion,
because the demonstration did.

Travel is the path length of the plan, summed along the chunk, for the joint
that moves most. The recorded chunk at the same frame is printed beside it.

    .\.venv-win\Scripts\python.exe tools\policy\does_it_stop.py [checkpoint ...]
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

EPISODES = int(os.environ.get("EPISODES", 20))
SAMPLES = int(os.environ.get("SAMPLES", 6))
MODELS = sys.argv[1:] or ["smolvla_v3only_20k", "smolvla_mix1_100k"]
DS_FOR = {"color6": "so101_color6", "c6pool": "so101_color6",
          "mix1": "so101_mix1", "v3": "so101_v3"}


def dataset_for(name: str) -> str:
    for key, ds in DS_FOR.items():
        if key in name:
            return ds
    raise SystemExit(f"{name}: cannot tell which dataset this was trained on")


def frame_at(ds, info, ep_row, key, t):
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
        last = None
        for f in c.decode(video=0):
            last = f
            if float(f.pts * s.time_base) >= t0 - 1e-6:
                break
        a = last.to_ndarray(format="rgb24").astype("float32") / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)


def travel(chunk):
    """Path length along the chunk, for the joint that moves most."""
    return float(np.abs(np.diff(chunk, axis=0)).sum(axis=0).max())


from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

for name in MODELS:
    ds = ROOT / "datasets" / dataset_for(name)
    # Some local copies carry data and meta but no videos. Say so now rather
    # than after eight minutes of measuring the first checkpoint.
    if not any(ds.glob("videos/**/*.mp4")):
        print(f"\n{name}: {ds.name} has no videos locally, skipping")
        continue
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    ep = pd.concat([pd.read_parquet(f) for f in
                    sorted((ds / "meta" / "episodes").rglob("*.parquet"))],
                   ignore_index=True).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in
                      sorted((ds / "data").rglob("*.parquet"))], ignore_index=True)
    tasks = pd.read_parquet(ds / "meta" / "tasks.parquet")
    task_of = {int(v): k for k, v in tasks["task_index"].items()}
    cache = {int(e): r.sort_values("frame_index") for e, r in data.groupby("episode_index")}
    chosen = sorted(cache)[::max(1, len(cache) // EPISODES)][:EPISODES]

    d = ROOT / "policies" / name
    policy = SmolVLAPolicy.from_pretrained(str(d)).to("cuda").eval()
    horizon = int(policy.config.chunk_size)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(d),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})

    out = {}
    for label, where in (("at the last frame", "end"), ("mid-episode", "mid")):
        planned, recorded = [], []
        for e in chosen:
            r = cache[e]
            i = len(r) - 1 if where == "end" else len(r) // 2
            t = float(r["timestamp"].to_numpy()[i])
            obs = {"observation.state": torch.tensor(
                       r["observation.state"].to_numpy()[i], dtype=torch.float32),
                   "task": task_of[int(r["task_index"].iloc[0])],
                   "robot_type": "so101_follower"}
            for cam, key in (("camera1", "observation.images.top"),
                             ("camera2", "observation.images.wrist")):
                obs[f"observation.images.{cam}"] = frame_at(ds, info, ep.loc[e], key, t)
            with torch.inference_mode():
                plans = []
                for _ in range(SAMPLES):
                    policy.reset()
                    plans.append(post(policy.predict_action_chunk(pre(dict(obs))))
                                 .squeeze(0).float().cpu().numpy())
            planned.append(np.mean([travel(p) for p in plans]))
            a = np.stack(r["action"].to_numpy()[i:i + horizon])
            recorded.append(travel(a))
        out[label] = (float(np.mean(planned)), float(np.mean(recorded)))

    print(f"\n{name}  ({dataset_for(name)}, {len(chosen)} episodes, "
          f"{SAMPLES} plans, chunk {horizon})")
    print(f"  {'':<20}{'policy plans':>14}{'demonstration did':>20}")
    for label, (p, rec) in out.items():
        print(f"  {label:<20}{p:14.1f}{rec:20.1f}")
    del policy
    torch.cuda.empty_cache()

print("\nTravel is how far the commanded pose moves over one chunk. At the last")
print("frame the demonstration is parked, so anything the policy plans there is")
print("motion it invented.")
