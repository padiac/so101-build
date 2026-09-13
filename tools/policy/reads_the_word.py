r"""Does the policy's plan change when the colour word changes?

This is the question the whole project has been circling, and two earlier attempts
at it were invalid: one compared normalized network output against unnormalized
joint angles, the other sampled the carry phase, where every instruction produces
the same motion.

This one is grounded differently. Flow matching draws fresh noise each time it
plans, so the SAME instruction already gives a spread of plans -- that spread is
the noise floor, measured here rather than assumed. The question then has a clear
form: does changing the colour word move the plan by more than the noise floor?

Same picture throughout, so the only thing that varies is the word.

    .\.venv-win\Scripts\python.exe tools\policy\reads_the_word.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAMPLES = 16
COLOURS = ["red", "blue", "yellow", "green", "purple"]
SENTENCE = "Pick the {} cube out of the bowl and put it on the table."

# Sentences the policy has never seen, including the opposite task and plain
# nonsense. If the plan does not move for these either, the language channel is
# dead outright and the colour result is a symptom rather than the disease.
WILD = {
    "opposite  ": "Pick the yellow block and put it in the black bowl.",
    "nonsense  ": "Bananas quietly compile the refrigerator on Tuesday.",
    "empty     ": "",
    "do nothing": "Do not move. Stay completely still.",
}
# Pass checkpoint names as arguments to compare any set; these are the default.
MODELS = sys.argv[1:] or ["smolvla_color6only_20k", "smolvla_mix1_100k"]
DS = ROOT / "datasets" / "so101_color6"


def frame_at(info, ep_row, key):
    import av                                    # after torch, or Windows segfaults

    v = DS / info["video_path"].format(
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

info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
ep = pd.concat([pd.read_parquet(f) for f in
                sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
               ignore_index=True).set_index("episode_index")
data = pd.concat([pd.read_parquet(f) for f in
                  sorted((DS / "data").rglob("*.parquet"))], ignore_index=True)
e = int(ep.index[0])
state = np.stack(data[data.episode_index == e]
                 .sort_values("frame_index")["observation.state"].to_numpy())[0]
images = {f"observation.images.{c}": frame_at(info, ep.loc[e], k)
          for c, k in (("camera1", "observation.images.top"),
                       ("camera2", "observation.images.wrist"))}

for name in MODELS:
    d = ROOT / "policies" / name
    policy = SmolVLAPolicy.from_pretrained(str(d)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(d),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})

    per_colour = {}
    with torch.inference_mode():
        for colour in COLOURS:
            obs = {"observation.state": torch.tensor(state, dtype=torch.float32),
                   "task": SENTENCE.format(colour),
                   "robot_type": "so101_follower", **images}
            plans = []
            for _ in range(SAMPLES):
                policy.reset()
                plans.append(post(policy.predict_action_chunk(pre(dict(obs))))
                             .squeeze(0).float().cpu().numpy())
            per_colour[colour] = np.stack(plans)[:, -1, 0]   # final shoulder_pan

    wild = {}
    with torch.inference_mode():
        for label, sentence in WILD.items():
            obs = {"observation.state": torch.tensor(state, dtype=torch.float32),
                   "task": sentence, "robot_type": "so101_follower", **images}
            plans = []
            for _ in range(SAMPLES):
                policy.reset()
                plans.append(post(policy.predict_action_chunk(pre(dict(obs))))
                             .squeeze(0).float().cpu().numpy())
            wild[label] = np.stack(plans)[:, -1, 0]

    means = np.array([per_colour[c].mean() for c in COLOURS])
    # The noise floor: how much the plan moves when nothing but the noise changes.
    within = float(np.mean([per_colour[c].std() for c in COLOURS]))
    between = float(means.std())

    print(f"\n{name}")
    print(f"  {SAMPLES} plans per colour, same picture, only the word changes")
    for c in COLOURS:
        v = per_colour[c]
        print(f"    {c:<8} shoulder_pan {v.mean():7.1f}  (sd {v.std():4.1f})")
    print(f"  spread BETWEEN colours : {between:5.2f}")
    print(f"  noise WITHIN a colour  : {within:5.2f}")
    print(f"  ratio                  : {between / within if within else 0:5.2f}"
          "   (1.0 means the word changes nothing)")
    base = float(np.mean([per_colour[c].mean() for c in COLOURS]))
    print("  sentences it was never trained on, against the colour average"
          f" ({base:.1f}):")
    for label, v in wild.items():
        print(f"    {label}  shoulder_pan {v.mean():7.1f}  (sd {v.std():4.1f})"
              f"   moved {abs(v.mean() - base):5.2f}")
    del policy
    torch.cuda.empty_cache()

print("\nThe word matters only if the between-colour spread stands well clear of")
print("the noise a single colour already produces on its own.")
