r"""Does the policy pick one of several targets, or average them?

Flow matching draws a fresh noise sample every time it plans, so a policy that has
learned "reach one of these six cubes" should produce visibly different plans from
the same picture -- one per cube. A policy that collapsed to the average produces
the same plan every time, aimed between them.

That distinction decides what to do next, and it is a property of the checkpoint,
so it needs no robot: take one real frame, plan from it many times, and look at
the spread.

Compared against a policy trained on a task with a single correct answer, which
is the control: its spread is what "no ambiguity" looks like on this rig.

    .\.venv-win\Scripts\python.exe tools\policy\sample_modes.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAMPLES = 24
# Same data, only the step count differs: a controlled look at whether longer
# training tightens the sampling, which is what would make the arm stop
# trembling. Everything else about these two is identical.
CASES = [
    ("smolvla_mix1_20k", "so101_v3", "so101_mix1, 20000 steps"),
    ("smolvla_mix1_100k", "so101_v3", "so101_mix1, 100000 steps -- only the steps differ"),
]


def frame_at(root: Path, info: dict, ep_row, offset_s: float, key: str):
    import av                                    # after torch, or Windows segfaults

    v = root / info["video_path"].format(
        video_key=key,
        chunk_index=int(ep_row[f"videos/{key}/chunk_index"]),
        file_index=int(ep_row[f"videos/{key}/file_index"]))
    t0 = float(ep_row[f"videos/{key}/from_timestamp"]) + offset_s
    with av.open(str(v)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s)
        for f in c.decode(video=0):
            if float(f.pts * s.time_base) >= t0 - 1e-6:
                a = f.to_ndarray(format="rgb24").astype("float32") / 255.0
                return torch.from_numpy(a).permute(2, 0, 1)
    raise RuntimeError(f"no frame at {t0:.2f}s in {v}")


from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

for name, dsname in [(c[0], c[1]) for c in CASES]:
    pass

for name, dsname, note in CASES:
    pol_dir = ROOT / "policies" / name
    ds = ROOT / "datasets" / dsname
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    ep = pd.concat([pd.read_parquet(f) for f in
                    sorted((ds / "meta" / "episodes").rglob("*.parquet"))],
                   ignore_index=True).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in
                      sorted((ds / "data").rglob("*.parquet"))], ignore_index=True)
    tasks = pd.read_parquet(ds / "meta" / "tasks.parquet")

    e = int(ep.index[0])
    g = data[data.episode_index == e].sort_values("frame_index")
    state = np.stack(g["observation.state"].to_numpy())[0]
    # The instruction the checkpoint was trained with, read from its own dataset.
    task = list(tasks.index)[0]

    policy = SmolVLAPolicy.from_pretrained(str(pol_dir)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(pol_dir),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})

    obs = {"observation.state": torch.tensor(state, dtype=torch.float32),
           "task": task, "robot_type": "so101_follower"}
    for cam, key in (("camera1", "observation.images.top"),
                     ("camera2", "observation.images.wrist")):
        obs[f"observation.images.{cam}"] = frame_at(ds, info, ep.loc[e], 0.0, key)

    plans = []
    with torch.inference_mode():
        for _ in range(SAMPLES):
            policy.reset()
            chunk = policy.predict_action_chunk(pre(dict(obs)))
            plans.append(post(chunk).squeeze(0).float().cpu().numpy())
    plans = np.stack(plans)                      # (samples, horizon, joints)

    end = plans[:, -1, :]                        # where each plan ends up
    print(f"\n{name}   [{note}]")
    print(f"  instruction: {task[:62]}")
    print(f"  {SAMPLES} plans from the SAME picture, spread of the final pose:")
    names = info["features"]["action"]["names"]
    for j, jn in enumerate(names[:5]):
        print(f"    {jn.split('.')[0]:<14} mean {end[:, j].mean():7.1f}   "
              f"sd {end[:, j].std():5.1f}   range {end[:, j].max() - end[:, j].min():6.1f}")
    del policy
    torch.cuda.empty_cache()

print("\nA policy that picks one of several targets shows a large spread here.")
print("A policy that averages them shows almost none, whatever it was asked.")
