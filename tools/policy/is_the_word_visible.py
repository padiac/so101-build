r"""Does the frozen backbone even represent the colour word differently?

Training freezes the vision-language model and updates only the action expert,
which is the recipe's own default. The open question is whether that freezing is
what stops the colour word from working. Two very different situations look the
same from outside:

  * the frozen features already separate "red" from "blue" clearly, and the
    action expert simply never learned to use them -- unfreezing adds capacity
    where the problem is the training signal, and would not help;
  * the frozen features barely differ between the colours, in which case no
    expert could use them however long it trains, and unfreezing is the fix.

This reads the prefix the expert is conditioned on -- the key/value cache the
frozen backbone produces from the picture, the state and the instruction -- and
measures how far apart the colours sit in it, against two yardsticks: sentences
that differ in far more than one word, and the same sentence twice.

    .\.venv-win\Scripts\python.exe tools\policy\is_the_word_visible.py [checkpoint]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DS = ROOT / "datasets" / "so101_color6"
COLOURS = ["red", "blue", "yellow", "green", "purple"]
SENTENCE = "Pick the {} cube out of the bowl and put it on the table."
FAR = ["Bananas quietly compile the refrigerator on Tuesday.",
       "Do not move. Stay completely still.",
       "Pick the yellow block and put it in the black bowl."]
CKPT = sys.argv[1] if len(sys.argv) > 1 else "smolvla_color6_453_100k"


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


info = json.loads((DS / "meta" / "info.json").read_text(encoding="utf-8"))
ep = pd.concat([pd.read_parquet(f) for f in
                sorted((DS / "meta" / "episodes").rglob("*.parquet"))],
               ignore_index=True).set_index("episode_index")
data = pd.concat([pd.read_parquet(f) for f in
                  sorted((DS / "data").rglob("*.parquet"))], ignore_index=True)
e = int(ep.index[0])
state = np.stack(data[data.episode_index == e].sort_values("frame_index")
                 ["observation.state"].to_numpy())[0]

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

d = ROOT / "policies" / CKPT
policy = SmolVLAPolicy.from_pretrained(str(d)).to("cuda").eval()
pre, _ = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=str(d),
    preprocessor_overrides={"device_processor": {"device": "cuda"}})
images = {f"observation.images.{c}": frame_at(info, ep.loc[e], k)
          for c, k in (("camera1", "observation.images.top"),
                       ("camera2", "observation.images.wrist"))}


def prefix(sentence):
    """The conditioning the action expert actually sees, as one flat vector."""
    obs = {"observation.state": torch.tensor(state, dtype=torch.float32),
           "task": sentence, "robot_type": "so101_follower", **images}
    batch = pre(dict(obs))
    with torch.inference_mode():
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        imgs, masks = policy.prepare_images(batch)
        st = policy.prepare_state(batch)
        lang = batch[OBS_LANGUAGE_TOKENS]
        lang_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
        embs, pad, att = policy.model.embed_prefix(imgs, masks, lang, lang_mask, state=st)
    return embs.float().flatten().cpu().numpy(), pad.flatten().cpu().numpy()


vecs = {c: prefix(SENTENCE.format(c))[0] for c in COLOURS}
far = [prefix(s)[0] for s in FAR]
same = prefix(SENTENCE.format("red"))[0]


def dist(a, b):
    return float(np.linalg.norm(a - b))


colour_pairs = [dist(vecs[a], vecs[b])
                for i, a in enumerate(COLOURS) for b in COLOURS[i + 1:]]
far_pairs = [dist(vecs["red"], f) for f in far]
repeat = dist(vecs["red"], same)

print(f"{CKPT}, same picture and state, only the sentence changes")
print(f"  the same sentence twice        : {repeat:8.2f}   (the floor)")
print(f"  one colour word swapped        : {np.mean(colour_pairs):8.2f}"
      f"   (min {min(colour_pairs):.2f}, max {max(colour_pairs):.2f})")
print(f"  a completely different sentence: {np.mean(far_pairs):8.2f}")
print()
if repeat < 1e-6:
    ratio = np.mean(colour_pairs) / max(np.mean(far_pairs), 1e-9)
    print(f"  colour swap as a share of a full sentence change: {ratio:.1%}")
print("\nThe frozen backbone is deterministic, so the floor should be zero. If a")
print("colour swap moves the conditioning a decent fraction of what a whole new")
print("sentence does, the information is present and freezing is not what stops")
print("the expert from using it.")
