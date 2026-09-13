r"""Does this policy's output actually change when the instruction changes?

The failure mode has a name -- instruction blindness -- and from the outside it is
indistinguishable from broken plumbing: in both cases the arm ignores what you
say. This separates them without touching the robot. Feed one real observation,
vary only the sentence, and measure how far apart the predicted action chunks are.

Identical chunks across instructions means the language input is dead. Chunks that
differ well above the noise floor mean the policy reads it, and any misbehaviour
on the bench is coming from somewhere else.

Frames are decoded straight from the mp4 rather than through LeRobotDataset,
whose decoder segfaults alongside a policy already resident on the GPU.

    .\.venv-win\Scripts\python.exe tools\policy\instruction_sensitivity.py ^
        policies\smolvla_mix1_100k datasets\so101_color6
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TASKS = [
    "Pick the red cube out of the bowl and put it on the table.",
    "Pick the blue cube out of the bowl and put it on the table.",
    "Pick the yellow cube out of the bowl and put it on the table.",
    "Pick the green cube out of the bowl and put it on the table.",
    "Pick the purple cube out of the bowl and put it on the table.",
    "Pick the yellow block and put it in the black bowl.",
]


def load_frame(ds_root: Path, info: dict, ep_row, ep_idx: int, offset_s: float, key: str):
    # PyAV is imported here, not at module level: loading its native FFmpeg
    # before torch's own native libraries segfaults this interpreter on Windows.
    import av

    v = ds_root / info["video_path"].format(
        video_key=key,
        chunk_index=int(ep_row[f"videos/{key}/chunk_index"]),
        file_index=int(ep_row[f"videos/{key}/file_index"]),
    )
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


def main() -> int:
    policy_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "policies/smolvla_mix1_100k")
    ds_root = Path(sys.argv[2] if len(sys.argv) > 2 else "datasets/so101_color6")
    n_obs = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(policy_dir))
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    info = json.loads((ds_root / "meta" / "info.json").read_text(encoding="utf-8"))
    ep = pd.concat(
        [pd.read_parquet(f) for f in sorted((ds_root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    ).set_index("episode_index")
    data = pd.concat(
        [pd.read_parquet(f) for f in sorted((ds_root / "data").rglob("*.parquet"))], ignore_index=True
    )

    print(f"policy   {policy_dir}")
    print(f"dataset  {ds_root}   {info['total_episodes']} episodes")
    print(f"testing  {len(TASKS)} instructions on {n_obs} observations\n", flush=True)

    step = max(len(ep) // n_obs, 1)
    eps = sorted(ep.index)[::step][:n_obs]

    rels = []
    for n, e in enumerate(eps):
        g = data[data.episode_index == e].sort_values("frame_index")
        mid = len(g) // 2
        state = torch.tensor(np.stack(g["observation.state"].to_numpy())[mid], dtype=torch.float32)
        row = ep.loc[e]
        top = load_frame(ds_root, info, row, int(e), mid / info["fps"], "observation.images.top")
        wrist = load_frame(ds_root, info, row, int(e), mid / info["fps"], "observation.images.wrist")

        chunks = []
        for task in TASKS:
            obs = {
                "observation.state": state,
                "observation.images.top": top,
                "observation.images.wrist": wrist,
                "task": task,
                "robot_type": "so101_follower",
            }
            policy.reset()
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(pre(dict(obs)))
            chunks.append(chunk.squeeze(0).float().cpu().numpy())

        C = np.stack(chunks)                       # (tasks, horizon, action_dim)
        between = np.abs(C[:, None] - C[None, :]).mean(axis=(2, 3))
        iu = np.triu_indices(len(TASKS), k=1)
        scale = float(np.abs(C).mean())
        spread = float(between[iu].mean())
        rel = spread / (scale + 1e-9)
        rels.append(rel)
        print(f"observation {n + 1}  (episode {int(e)}, mid-frame)")
        print(f"  mean |action|              {scale:.4f}")
        print(f"  mean pairwise difference   {spread:.4f}")
        print(f"  relative                   {rel:.1%}", flush=True)

    avg = float(np.mean(rels))
    print(f"\naverage relative spread between instructions: {avg:.1%}")
    if avg < 0.02:
        print("  -> the instruction changes almost nothing: the language input is dead.")
        print("     This is instruction blindness, not a wiring problem.")
    elif avg < 0.10:
        print("  -> weak. The instruction registers but barely steers the action.")
    else:
        print("  -> the policy clearly reads the instruction. Misbehaviour on the bench")
        print("     is coming from somewhere other than language.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
