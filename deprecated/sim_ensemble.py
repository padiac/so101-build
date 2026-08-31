"""With the arm stuck, what command does the ensembler actually produce?

At the freeze the individual chunks are unanimous -- shoulder_lift +87.6 with a
standard deviation of 1.9 over 40 consecutive observations -- yet the arm holds
still. So the chunks are not cancelling; the ensembling is turning them into a
near-stationary command.

While the arm is stuck the observation barely changes, so this reproduces that
state directly: hold one frozen-window observation fixed, run the ensembler for
100 ticks, and measure how far the commanded position travels. Sweeping the
coefficient shows which setting escapes.

lerobot's convention (modeling_act.py:171): w_i = exp(-coeff * i) with w_0 the
OLDEST prediction, so positive weights old predictions more and negative
weights new ones more.

ASCII output only.
"""

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

CK = "policies/act_trim_100k"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FROZEN_FRAME = 320
TICKS = 100

policy = ACTPolicy.from_pretrained(CK).to(DEV).eval()
cfg = PreTrainedConfig.from_pretrained(CK)
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CK,
    preprocessor_overrides={"device_processor": {"device": DEV}})
roll = LeRobotDataset("local/roll", root="datasets/rollout_probe")

s = roll[FROZEN_FRAME]
batch = {k: s[k][None] for k in
         ("observation.state", "observation.images.top", "observation.images.wrist")}

# one chunk from that observation, reused every tick (the observation is static
# while the arm is stuck, so every tick would produce this same chunk)
with torch.no_grad():
    policy.reset()
    o = pre(batch)
    chunk = torch.stack([policy.select_action(o)[0] for _ in range(cfg.chunk_size)])
chunk_np = chunk.cpu().numpy()
print("frozen frame %d; a single chunk spans %.1f units on shoulder_lift"
      % (FROZEN_FRAME, float(chunk_np[:, 1].max() - chunk_np[:, 1].min())))
print()

print("%12s %18s %18s" % ("coeff", "cmd travel (sh_lift)", "cmd travel (elbow)"))
for coeff in (0.05, 0.01, 0.0, -0.01, -0.05, -0.2):
    ens = ACTTemporalEnsembler(coeff, cfg.chunk_size)
    ens.reset()
    out = []
    for _ in range(TICKS):
        a = ens.update(chunk[None].cpu())        # (1, chunk, 6)
        out.append(a[0].numpy())
    out = np.array(out)
    print("%12.3f %18.2f %18.2f"
          % (coeff,
             out[:, 1].max() - out[:, 1].min(),
             out[:, 2].max() - out[:, 2].min()))
print()
print("travel near zero means that setting keeps the command pinned at the")
print("start of the ramp no matter how many ticks pass.")
