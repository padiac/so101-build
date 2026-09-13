r"""Where does the trembling come from: the replan, or the plan itself?

An earlier look measured the spread of the plan's FINAL pose and found the 100k
model more spread than the 20k one -- the opposite of how they feel on the arm, so
that number is not what makes it tremble.

Two things could. The arm executes a whole chunk and then replans, so a fresh plan
that disagrees with the old one AT ITS FIRST STEP yanks the arm the moment the
queue refills. And a plan can be internally rough, which shakes the arm all the
way through it. These are different faults with different fixes, and both are
properties of the checkpoint.

    .\.venv-win\Scripts\python.exe tools\policy\tremor.py [checkpoint ...]
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

SAMPLES = int(os.environ.get("SAMPLES", 16))
DEFAULT = ["smolvla_mix1_20k", "smolvla_mix1_100k", "smolvla_v3only_20k",
           "smolvla_color6_304_20k"]


def resolve(arg: str) -> Path:
    """A published name under policies/, or a path to any checkpoint directory.

    Every intermediate checkpoint of a run is kept, so the roughness curve over
    training can be measured without training anything again.
    """
    d = ROOT / "policies" / arg
    if (d / "config.json").exists():
        return d
    d = Path(arg)
    if (d / "config.json").exists():
        return d
    raise SystemExit(f"{arg}: no checkpoint here")


def dataset_for(name: str) -> str:
    """Which dataset supplies the picture and the instruction.

    A lookup table silently sent any unlisted name to so101_v3, which would
    compare a colour model against the wrong scene. Derive it from the name
    instead, so a new checkpoint cannot be measured against the wrong pictures.
    """
    if "color6" in name or "c6pool" in name:
        return "so101_color6"
    if "v3" in name or "mix1" in name:
        return "so101_v3"
    raise SystemExit(f"{name}: cannot tell which dataset this was trained on")


def frame_at(ds: Path, info: dict, ep_row, key: str):
    import av                                    # after torch, or Windows segfaults

    v = ds / info["video_path"].format(
        video_key=key,
        chunk_index=int(ep_row[f"videos/{key}/chunk_index"]),
        file_index=int(ep_row[f"videos/{key}/file_index"]))
    t0 = float(ep_row[f"videos/{key}/from_timestamp"])
    with av.open(str(v)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s)
        for f in c.decode(video=0):
            if float(f.pts * s.time_base) >= t0 - 1e-6:
                a = f.to_ndarray(format="rgb24").astype("float32") / 255.0
                return torch.from_numpy(a).permute(2, 0, 1)
    raise RuntimeError("no frame")


from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

for name in (sys.argv[1:] or DEFAULT):
    ckpt = resolve(name)
    ds = ROOT / "datasets" / dataset_for(name.replace("\\", "/"))
    info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    ep = pd.concat([pd.read_parquet(f) for f in
                    sorted((ds / "meta" / "episodes").rglob("*.parquet"))],
                   ignore_index=True).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in
                      sorted((ds / "data").rglob("*.parquet"))], ignore_index=True)
    e = int(ep.index[0])
    state = np.stack(data[data.episode_index == e]
                     .sort_values("frame_index")["observation.state"].to_numpy())[0]
    task = list(pd.read_parquet(ds / "meta" / "tasks.parquet").index)[0]

    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})

    obs = {"observation.state": torch.tensor(state, dtype=torch.float32),
           "task": task, "robot_type": "so101_follower"}
    for cam, key in (("camera1", "observation.images.top"),
                     ("camera2", "observation.images.wrist")):
        obs[f"observation.images.{cam}"] = frame_at(ds, info, ep.loc[e], key)

    plans = []
    with torch.inference_mode():
        for _ in range(SAMPLES):
            policy.reset()
            plans.append(post(policy.predict_action_chunk(pre(dict(obs))))
                         .squeeze(0).float().cpu().numpy())
    P = np.stack(plans)                          # (samples, horizon, joints)
    H = P.shape[1]

    # Disagreement between plans, at each point along the chunk. What matters for
    # a yank is the value at step 1: that is where a fresh plan takes over.
    spread = P.std(axis=0).mean(axis=1)
    # Roughness inside one plan: how much the commanded position reverses
    # direction from step to step, averaged over plans and joints. A smooth reach
    # is near zero; a shaky one is not.
    d = np.diff(P, axis=1)
    # Per plan, so the spread across plans gives the measurement its own error
    # bar. Two runs of this script disagreed by 0.10 on the same checkpoint, so a
    # single number cannot say whether two checkpoints differ.
    per_plan = np.abs(np.diff(d, axis=1)).mean(axis=(1, 2))
    rough = float(per_plan.mean())
    rough_se = float(per_plan.std(ddof=1) / np.sqrt(len(per_plan)))

    print(f"\n{name}")
    print(f"  disagreement between plans, along the chunk (units):")
    marks = [0, 1, 2, H // 8, H // 4, H // 2, 3 * H // 4, H - 1]
    print("    " + "  ".join(f"step{m + 1:>3}" for m in marks))
    print("    " + "  ".join(f"{spread[m]:7.1f}" for m in marks))
    print(f"  yank at the replan (step 1 disagreement): {spread[0]:5.2f}")
    print(f"  roughness inside one plan               : {rough:5.3f}"
          f"  +/- {rough_se:.3f}  ({SAMPLES} plans)")
    del policy
    torch.cuda.empty_cache()

print("\nA yank at step 1 shakes the arm every time the chunk refills, about every")
print("1.7 s. Roughness shakes it continuously, all the way through the plan.")
