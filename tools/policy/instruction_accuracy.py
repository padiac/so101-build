r"""Does the policy pick the cube it was told to, or just something?

instruction_sensitivity.py answers the weaker question -- whether the output moves
at all when the sentence changes. It can move a lot and still be wrong. This asks
the question that matters: given a real observation whose correct instruction is
known, feed all of them and see whether the true one produces the action closest
to what the demonstrator actually did.

Scoring is retrieval, not regression: for each held-out frame, rank the six
instructions by how far their predicted chunk sits from the recorded chunk, and
count how often the true instruction ranks first. Chance is 1/6 = 16.7%. Well
above chance means the language is genuinely steering the reach; at chance the
policy is picking a cube for reasons unrelated to what it was asked.

Frames are decoded straight from the mp4: LeRobotDataset's decoder segfaults
alongside a policy already resident on the GPU.

    .\.venv-win\Scripts\python.exe tools\policy\instruction_accuracy.py ^
        policies\smolvla_mix1_100k datasets\so101_color6 30
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_frame(ds_root: Path, info: dict, ep_row, offset_s: float, key: str):
    # PyAV is imported here, not at module level: loading its native FFmpeg before
    # torch's own native libraries segfaults this interpreter on Windows.
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
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(policy_dir))
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    horizon = int(policy.config.chunk_size)

    info = json.loads((ds_root / "meta" / "info.json").read_text(encoding="utf-8"))
    ep = pd.concat(
        [pd.read_parquet(f) for f in sorted((ds_root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    ).set_index("episode_index")
    data = pd.concat(
        [pd.read_parquet(f) for f in sorted((ds_root / "data").rglob("*.parquet"))], ignore_index=True
    )
    tasks_df = pd.read_parquet(ds_root / "meta" / "tasks.parquet")
    idx2task = {int(v): k for k, v in zip(tasks_df.index, tasks_df.task_index)}
    TASKS = [idx2task[i] for i in sorted(idx2task)]

    print(f"policy   {policy_dir}")
    print(f"dataset  {ds_root}   {info['total_episodes']} episodes, {len(TASKS)} instructions")
    print(f"chance   {1 / len(TASKS):.1%}\n", flush=True)

    rng = np.random.default_rng(0)
    eps = rng.choice(sorted(ep.index), size=min(n_trials, len(ep)), replace=False)

    hits, ranks, per_task = 0, [], {}
    confusion = np.zeros((len(TASKS), len(TASKS)))
    for n, e in enumerate(eps):
        g = data[data.episode_index == int(e)].sort_values("frame_index")
        true_task = idx2task[int(g.task_index.iloc[0])]
        A = np.stack(g["action"].to_numpy())
        S = np.stack(g["observation.state"].to_numpy())
        # Sample from the reaching phase: the first third is often still setup and
        # the last third is the carry, where every instruction looks the same.
        t = len(g) // 2
        if t + horizon >= len(g):
            continue
        truth = A[t : t + horizon]

        row = ep.loc[int(e)]
        obs_common = {
            "observation.state": torch.tensor(S[t], dtype=torch.float32),
            "observation.images.top": load_frame(ds_root, info, row, t / info["fps"],
                                                 "observation.images.top"),
            "observation.images.wrist": load_frame(ds_root, info, row, t / info["fps"],
                                                   "observation.images.wrist"),
            "robot_type": "so101_follower",
        }

        errs = []
        for task in TASKS:
            policy.reset()
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(pre({**obs_common, "task": task}))
            pred = chunk.squeeze(0).float().cpu().numpy()[: len(truth)]
            errs.append(float(np.abs(pred - truth).mean()))

        order = np.argsort(errs)
        true_i = TASKS.index(true_task)
        rank = int(np.where(order == true_i)[0][0]) + 1
        ranks.append(rank)
        hit = rank == 1
        hits += hit
        d = per_task.setdefault(true_task, [0, 0])
        d[0] += hit
        d[1] += 1
        confusion[true_i, int(order[0])] += 1
        if n < 8:
            short = true_task.split()[2]
            print(f"  ep {int(e):>3}  true={short:<7} rank {rank}/{len(TASKS)}  "
                  f"err(true) {errs[true_i]:.4f}  err(best) {min(errs):.4f}")

    n_done = len(ranks)
    acc = hits / max(n_done, 1)
    chance = 1 / len(TASKS)

    # Binomial test against chance, because the per-instruction split is far too
    # thin to read: thirty trials over five instructions is six apiece, and six
    # coin flips look like a pattern often enough to fool anyone.
    p = chance
    sd = (n_done * p * (1 - p)) ** 0.5
    z = (hits - n_done * p) / sd if sd else 0.0

    print(f"\n{n_done} trials")
    print(f"  true instruction ranked first : {hits}/{n_done} = {acc:.1%}   (chance {chance:.1%})")
    print(f"  mean rank of true instruction : {np.mean(ranks):.2f} / {len(TASKS)}")
    print(f"  z against chance              : {z:.2f}" + ("   (significant)" if abs(z) > 2 else "   (not significant)"))

    short = [t.split()[2] for t in TASKS]
    print("\n  confusion -- rows: what it was told, columns: which instruction best")
    print("  explains the action it produced. A diagonal means it follows the word;")
    print("  a hot column means it does the same thing whatever you say.")
    print("\n      " + "".join(f"{s[:6]:>8}" for s in short))
    for i, s in enumerate(short):
        row = confusion[i]
        tot = row.sum()
        cells = "".join(f"{(row[j] / tot if tot else 0):>7.0%} " for j in range(len(TASKS)))
        print(f"  {s[:5]:<5} {cells}  n={int(tot)}")

    col = confusion.sum(axis=0)
    worst = int(np.argmax(col))
    print(f"\n  most-chosen instruction overall: {short[worst]} "
          f"({col[worst] / col.sum():.0%} of all trials)")

    print()
    if abs(z) <= 2:
        print("  -> not distinguishable from chance at this sample size.")
    elif acc < 0.45:
        print("  -> better than chance, but wrong most of the time. The instruction")
        print("     shifts the behaviour without reliably selecting the target.")
    else:
        print("  -> the instruction genuinely selects the target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
