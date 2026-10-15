r"""Put the policy where the demonstrator was about to close. Does its plan close?

On the robot color6_304_100k closed the gripper in 10/10 trials and
color6_453_40k in 1/10. Those two differ in three things at once -- data (304 vs
453 episodes), steps (100k vs 40k) and LR schedule -- so the robot result cannot
say which. The 304 run kept every checkpoint, and its 40k one shares the data of
the first model and the step count of the second.

Testing that on the robot costs twenty minutes a checkpoint. This is the cheap
version, and it is only allowed to answer if it passes two checks first:

  1. The demonstration's own next actions, scored by the same rule, must close.
  2. It must reproduce the robot: 304_100k closes, 453_40k does not.

If it fails either, it measures something else and says so instead of a verdict.

Closure rule, same spirit as rubric_metrics.closures: within the 50-step plan the
gripper command falls at least DROP units below its highest point so far, and
stays there for HOLD steps.

    .\.venv-win\Scripts\python.exe tools\policy\will_it_close.py CKPT [CKPT ...]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "policy"))
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
import box_cubes as B                        # noqa: E402
from aimed_at import grasp_frame             # noqa: E402

DS = ROOT / "datasets" / "so101_color6"
EPISODES = int(os.environ.get("EPISODES", 20))
SAMPLES = int(os.environ.get("SAMPLES", 4))
LEADS = (9, 30)          # frames before the demonstrated closure: 0.3 s and 1 s
DROP, HOLD = 8.0, 6


def closes(g: np.ndarray) -> bool:
    top = np.maximum.accumulate(g)
    low = g < top - DROP
    run = 0
    for v in low:
        run = run + 1 if v else 0
        if run >= HOLD:
            return True
    return False


B.DS = DS
info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
GRIP = info["features"]["action"]["names"].index("gripper.pos")
ep = pd.concat([pd.read_parquet(f) for f in sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
               ignore_index=True).set_index("episode_index")
data = pd.concat([pd.read_parquet(f) for f in sorted((DS / "data").rglob("*.parquet"))], ignore_index=True)
tasks = pd.read_parquet(DS / "meta" / "tasks.parquet")
task_of = {int(v): k for k, v in tasks["task_index"].items()}
cache = {int(e): r.sort_values("frame_index") for e, r in data.groupby("episode_index")}

points = []                                  # (episode, frame, lead)
for e in sorted(cache)[:: max(1, len(cache) // EPISODES)][:EPISODES]:
    act = np.stack(cache[e]["action"].to_numpy())
    j = grasp_frame(act[:, GRIP])
    if j is None:
        continue
    for lead in LEADS:
        if j - lead >= 0 and j - lead + 50 <= len(act):
            points.append((e, j - lead, lead))

print(f"{len(points)} moments from {len({p[0] for p in points})} demonstrations, "
      f"{SAMPLES} plans each\n")
demo = {lead: np.mean([closes(np.stack(cache[e]["action"].to_numpy())[f:f + 50, GRIP])
                       for e, f, l in points if l == lead]) for lead in LEADS}
print("check 1 -- the demonstration's own next 50 actions close:")
for lead in LEADS:
    print(f"  {lead / 30:.1f} s before the grasp: {demo[lead]:.0%}")

IM = {}
for e, f, _ in points:
    t = float(cache[e]["timestamp"].to_numpy()[f])
    IM[(e, f)] = {f"observation.images.{c}": torch.from_numpy(
        B.frame_at(info, ep.loc[e], k, t).astype("float32") / 255.0).permute(2, 0, 1)
        for c, k in (("camera1", "observation.images.top"), ("camera2", "observation.images.wrist"))}

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("\n  checkpoint" + " " * 42 + "  ".join(f"{l / 30:.1f}s before" for l in LEADS))
for arg in sys.argv[1:]:
    d = ROOT / "policies" / arg if (ROOT / "policies" / arg / "config.json").exists() else Path(arg)
    policy = SmolVLAPolicy.from_pretrained(str(d)).to("cuda").eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(d),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    rate = {lead: [] for lead in LEADS}
    for e, f, lead in points:
        r = cache[e]
        obs = {"observation.state": torch.tensor(r["observation.state"].to_numpy()[f], dtype=torch.float32),
               "task": task_of[int(r["task_index"].iloc[0])], "robot_type": "so101_follower", **IM[(e, f)]}
        with torch.inference_mode():
            for _ in range(SAMPLES):
                policy.reset()
                chunk = post(policy.predict_action_chunk(pre(dict(obs)))).squeeze(0).float().cpu().numpy()
                rate[lead].append(closes(chunk[:, GRIP]))
    label = arg if len(arg) < 50 else "..." + arg[-47:]
    print(f"  {label:<50}" + "  ".join(f"{np.mean(rate[l]):>11.0%}" for l in LEADS))
    del policy
    torch.cuda.empty_cache()

print("\ncheck 2 is read off the table: on the robot 304_100k closed 10/10 and")
print("453_40k 1/10. If the two rows above do not differ the same way, this is not")
print("measuring what failed on the robot, and no row here should be believed.")
