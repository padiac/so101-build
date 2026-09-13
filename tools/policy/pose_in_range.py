r"""Is this start pose inside anything the checkpoint has ever seen?

Homing to another task's start pose has now gone wrong twice, both times
silently, both times in wrist_roll, and both times it looked like the policy was
bad at grasping. Matching dataset names is what failed: the training copy is
called so101_color6_304_trim and the local one so101_color6, so the lookup missed
and fell back to so101_v2, 84 units away in a joint the wrist camera is bolted
to.

Names are the wrong thing to check. Every checkpoint carries the statistics of
its own training data in its normalizer, so ask the checkpoint directly. A joint
outside the min..max it was trained on is a pose no demonstration ever contained,
whatever the datasets are called.

Exit 1 if any joint is outside, so a caller can refuse to run.

    .\.venv-win\Scripts\python.exe tools\policy\pose_in_range.py POLICY DATASET
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KEY = "observation.state"


def stats(policy_dir: Path) -> dict:
    """min/max/q01/q99 the checkpoint recorded for its own training states."""
    from safetensors.torch import load_file

    files = sorted(policy_dir.glob("*normalizer_processor.safetensors"))
    for f in files:
        d = load_file(str(f))
        if f"{KEY}.min" in d:
            return {k.split(".")[-1]: d[k].flatten().numpy()
                    for k in d if k.startswith(KEY + ".")}
    raise SystemExit(f"{policy_dir.name}: no {KEY} statistics in the checkpoint")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    policy, dataset = Path(argv[0]), Path(argv[1])
    if not policy.is_absolute():
        policy = ROOT / policy
    if not dataset.is_absolute():
        dataset = ROOT / dataset

    sys.path.insert(0, str(ROOT))
    import home as home_mod

    target, _ = home_mod.target_pose(str(dataset))
    s = stats(policy)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"][KEY]["names"]
    # The checkpoint stores no joint names, and neither does lerobot when it
    # normalizes at inference -- both are positional. Pairing them any other way
    # would be inventing an alignment the runtime does not use.
    if len(names) != len(s["min"]):
        print(f"[X] {len(names)} joints here, {len(s['min'])} in the checkpoint")
        return 1

    bad = []
    print(f"{policy.name} vs the start pose of {dataset.name}")
    print(f"  {'joint':<18}{'target':>8}{'trained range':>22}")
    for j, n in enumerate(names):
        lo, hi = float(s["min"][j]), float(s["max"][j])
        v = float(target[j])
        out = v < lo or v > hi
        mark = "   <-- OUTSIDE" if out else ""
        print(f"  {n:<18}{v:8.1f}{lo:11.1f} ..{hi:8.1f}{mark}")
        if out:
            bad.append((n, v, lo, hi))

    if bad:
        print()
        for n, v, lo, hi in bad:
            off = max(lo - v, v - hi)
            print(f"[X] {n} would start {off:.0f} units outside anything this "
                  f"checkpoint was trained on")
        print("[X] this is not the data this policy was trained on")
        return 1
    print("[OK] every joint starts inside the checkpoint's own training range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
