#!/usr/bin/env python3
"""Fix lerobot's ACT crashing when the policy has no proprioceptive input.

ACT supports running without observation.state -- every use of
robot_state_feature in modeling_act.py is guarded. One line is not:

    latent_sample = torch.zeros([...]).to(batch[OBS_STATE].device)

in the use_vae=False branch, where the state tensor is read purely to obtain a
device. With no state feature in the dataset that raises

    KeyError: 'observation.state'

Take the device from the module's own weights instead, which is what was meant.

Why we run without state at all: the proprioceptive input lets the policy
memorise trajectories keyed on the arm's pose rather than locate the object in
the image. Measured on the v2 policy, swapping the camera images moved its plan
by 0.85 while the real targets spanned 16.23 -- it was not looking. See
https://arxiv.org/html/2509.18644v1 (Do You Need Proprioceptive States in
Visuomotor Policies?).

Usage:
    python patch_act_nostate.py <path-to-site-packages>
"""

import io
import sys
from pathlib import Path

OLD = """            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )"""

NEW = """            # Device taken from the module's own weights, not from
            # batch[OBS_STATE]: a state-free policy has no such key, and this
            # line only ever wanted a device. (patch_act_nostate.py)
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                self.encoder_latent_input_proj.weight.device
            )"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    p = Path(sys.argv[1]) / "lerobot" / "policies" / "act" / "modeling_act.py"
    if not p.exists():
        print("[X] not found: %s" % p)
        return 1
    s = io.open(p, encoding="utf-8").read()
    if "patch_act_nostate.py" in s:
        print("[OK] already patched: %s" % p)
        return 0
    if OLD not in s:
        print("[X] the expected code was not found in %s" % p)
        print("    lerobot may have changed; inspect the use_vae=False branch by hand.")
        return 1
    io.open(str(p) + ".bak", "w", encoding="utf-8", newline="\n").write(s)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(OLD, NEW, 1))
    print("[OK] patched %s  (backup at .bak)" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
