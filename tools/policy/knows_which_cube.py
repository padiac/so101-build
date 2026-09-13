r"""Does the policy plan for THE cube in front of it, or for cubes in general?

Every earlier language test held the picture fixed and changed the word. That
asks a fair question but a narrow one, and it has an answer already: the colour
word moves the plan less than the model's own sampling noise.

This asks the question the task actually poses. Six cubes sit in the bowl in
every frame, and which one to take is fixed only by the word AND the picture
together -- the word alone cannot say where to reach, which is why the colour a
demonstration names explains almost none of where its arm ends up. So: shown one
real moment from one real episode, does the model plan that episode's own next
move, or something generic that fits any episode equally?

The recorded future is the ground truth, so no landmark detector and no robot
are needed. For each episode the plan is scored against its own recorded future
and against every other episode's future at the same point. Chance is the 50th
percentile; grounding means the model's own future ranks near the top.

Two things keep the score honest.

The moment to ask at is not chosen by eye: it is the point where the recorded
futures disagree with each other the most, measured from the data before the
model is loaded. Asking earlier would be asking while every demonstration is
still doing the same thing, and any policy would score at chance through no
fault of its own.

And the arm is already moving by then, so its joint angles alone may say where
it is going. A control with no model in it measures exactly that, and each
condition below keeps the joint angles and changes one other thing -- a score
that survives the change was not coming from that channel.

    .\.venv-win\Scripts\python.exe tools\policy\knows_which_cube.py [checkpoint ...]
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

DS = ROOT / "datasets" / "so101_color6"
HORIZON = 50                       # the policy's chunk
EPISODES = int(os.environ.get("EPISODES", 40))
SAMPLES = int(os.environ.get("SAMPLES", 6))
MODELS = sys.argv[1:] or ["smolvla_color6_304_100k"]


def frame_at(info, ep_row, key, t):
    import av                                    # after torch, or Windows segfaults

    v = DS / info["video_path"].format(
        video_key=key,
        chunk_index=int(ep_row[f"videos/{key}/chunk_index"]),
        file_index=int(ep_row[f"videos/{key}/file_index"]))
    t0 = float(ep_row[f"videos/{key}/from_timestamp"]) + t
    with av.open(str(v)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s)
        for f in c.decode(video=0):
            if float(f.pts * s.time_base) >= t0 - 1e-6:
                a = f.to_ndarray(format="rgb24").astype("float32") / 255.0
                return torch.from_numpy(a).permute(2, 0, 1)
    raise RuntimeError("no frame")


info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
ep = pd.concat([pd.read_parquet(f) for f in
                sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
               ignore_index=True).set_index("episode_index")
data = pd.concat([pd.read_parquet(f) for f in
                  sorted((DS / "data").rglob("*.parquet"))], ignore_index=True)
tasks = pd.read_parquet(DS / "meta" / "tasks.parquet")
task_of = {int(v): k for k, v in tasks["task_index"].items()}

# One episode in every colour, evenly through the recording order, so a drift in
# how the cubes were laid out over the session cannot masquerade as a result.
by_colour = {}
for e, rows in data.groupby("episode_index"):
    by_colour.setdefault(int(rows["task_index"].iloc[0]), []).append(int(e))
chosen = []
for ti, eps in sorted(by_colour.items()):
    take = max(1, EPISODES // len(by_colour))
    chosen += sorted(eps)[::max(1, len(eps) // take)][:take]
chosen = sorted(chosen)
cache = {e: data[data.episode_index == e].sort_values("frame_index")
         for e in chosen}

# Where in the episode do the recorded futures disagree most? Measured from the
# data alone, before any model is loaded.
futures_at = {}
for frac in np.arange(0.10, 0.85, 0.05):
    F = []
    for e in chosen:
        r = cache[e]
        i = int(round(frac * len(r)))
        if i + HORIZON > len(r):
            F = []
            break
        F.append(np.stack(r["action"].to_numpy()[i:i + HORIZON]))
    if F:
        futures_at[round(float(frac), 2)] = np.stack(F)

spread = {f: float(np.mean(np.std(F, axis=0))) for f, F in futures_at.items()}
FRAC = max(spread, key=spread.get)
print("how much the recorded futures disagree, through the episode:")
print("  " + "  ".join(f"{f:.2f}" for f in sorted(spread)))
print("  " + "  ".join(f"{spread[f]:4.1f}" for f in sorted(spread)))
print(f"asking at {FRAC:.0%} of the episode, where they disagree most "
      f"({spread[FRAC]:.1f} units)")
print(f"{len(chosen)} episodes, {len(by_colour)} colours, {SAMPLES} plans each")

FUT = futures_at[FRAC]                               # (episodes, horizon, joints)
AT = {e: int(round(FRAC * len(cache[e]))) for e in chosen}
STATE = np.stack([np.asarray(cache[e]["observation.state"].to_numpy()[AT[e]],
                             dtype=np.float64) for e in chosen])


def score(plans, label):
    """Rank each episode's own recorded future against all the others."""
    ranks, gaps = [], []
    for k, plan in enumerate(plans):
        dist = np.sqrt(((FUT - plan[None]) ** 2).mean(axis=(1, 2)))
        own, others = dist[k], np.delete(dist, k)
        ranks.append(float((others > own).mean()))        # 1.0 = own fits best
        gaps.append(float(others.mean() - own))
    print(f"  {label:<24} beats {np.mean(ranks):5.1%},  best in "
          f"{np.mean(np.array(ranks) == 1.0):5.1%} of episodes,  "
          f"{np.mean(gaps):6.2f} units closer")
    return float(np.mean(ranks))


# The control with no model in it: nearest neighbours in joint space, the
# episode itself excluded. If this already scores high, the joint angles say
# where the arm is going and the model's score is not about the cube.
d2 = ((STATE[:, None, :] - STATE[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(d2, np.inf)
knn = np.argsort(d2, axis=1)[:, :3]
print("\ncontrol, no model:")
score(FUT[knn].mean(axis=1), "joint angles alone")

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Decode once; the ablations re-use the same frames.
IMAGES, TASK, WRONG = [], [], []
for e in chosen:
    t = float(cache[e]["timestamp"].to_numpy()[AT[e]])
    IMAGES.append({f"observation.images.{cam}": frame_at(info, ep.loc[e], key, t)
                   for cam, key in (("camera1", "observation.images.top"),
                                    ("camera2", "observation.images.wrist"))})
    ti = int(cache[e]["task_index"].iloc[0])
    TASK.append(task_of[ti])
    WRONG.append(task_of[(ti + 2) % len(task_of)])

for name in MODELS:
    d = ROOT / "policies" / name
    policy = SmolVLAPolicy.from_pretrained(str(d)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(d),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})

    def plan_for(k, images, task):
        e = chosen[k]
        obs = {"observation.state": torch.tensor(
                   cache[e]["observation.state"].to_numpy()[AT[e]],
                   dtype=torch.float32),
               "task": task, "robot_type": "so101_follower", **images}
        with torch.inference_mode():
            plans = []
            for _ in range(SAMPLES):
                policy.reset()
                plans.append(post(policy.predict_action_chunk(pre(dict(obs))))
                             .squeeze(0).float().cpu().numpy())
        return np.stack(plans).mean(axis=0)

    n = len(chosen)

    def one_camera_from(k, j, cam):
        """Episode k's view with a single camera taken from episode j."""
        d = dict(IMAGES[k])
        d[f"observation.images.{cam}"] = IMAGES[j][f"observation.images.{cam}"]
        return d

    print(f"\n{name}")
    score([plan_for(k, IMAGES[k], TASK[k]) for k in range(n)], "as recorded")
    score([plan_for(k, IMAGES[(k + 1) % n], TASK[k]) for k in range(n)],
          "both views swapped")
    # Separately, because they carry different things: the top view sees the
    # whole bowl, the wrist view sees whatever the gripper is pointed at.
    score([plan_for(k, one_camera_from(k, (k + 1) % n, "camera1"), TASK[k])
           for k in range(n)], "only the top view swapped")
    score([plan_for(k, one_camera_from(k, (k + 1) % n, "camera2"), TASK[k])
           for k in range(n)], "only the wrist view swapped")
    score([plan_for(k, IMAGES[k], WRONG[k]) for k in range(n)],
          "the wrong colour word")
    del policy
    torch.cuda.empty_cache()

print("\nAt chance (50%) the plan fits any episode as well as the one it was")
print("shown. A condition scoring as high as 'as recorded' is a channel the")
print("model was not relying on.")
