r"""Would this dataset actually train against this checkpoint? Answer in seconds.

On 2026-09-05 a run merged and trimmed for ninety minutes, then died in the first
second of training because `lerobot/smolvla_base` names its cameras camera1/2/3
and the dataset names them top/wrist. Nothing about that needed the GPU, the
merge, or the trim to discover.

So: check the cheap, deterministic things first -- feature names, shapes, action
dimension -- and only then spend the ninety minutes.

    .\.venv-win\Scripts\python.exe preflight_train.py datasets\so101_v3 --policy survey\models\smolvla_base
"""

import argparse
import json
import sys
from pathlib import Path


def load_policy_features(policy_dir: Path) -> dict:
    cfg = json.loads((policy_dir / "config.json").read_text(encoding="utf-8"))
    return cfg.get("input_features", {}), cfg.get("output_features", {})


def load_dataset_features(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return info["features"], info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--policy", type=Path, required=True, help="local dir of the pretrained checkpoint")
    ap.add_argument("--rename", default="", help='JSON map applied to the dataset keys, as lerobot-train takes it')
    args = ap.parse_args()

    pin, pout = load_policy_features(args.policy)
    dfeat, info = load_dataset_features(args.dataset)
    rename = json.loads(args.rename) if args.rename else {}

    p_vis = sorted(k for k, v in pin.items() if v.get("type") == "VISUAL")
    d_vis = sorted(rename.get(k, k) for k, v in dfeat.items() if v.get("dtype") == "video")

    print(f"dataset  {args.dataset}   {info['total_episodes']} episodes / {info['total_frames']} frames")
    print(f"policy   {args.policy}")
    print()
    print(f"policy expects visuals : {p_vis}")
    print(f"dataset provides       : {d_vis}" + ("   (after rename)" if rename else ""))

    ok = True
    # lerobot accepts either direction as a subset; the empty-camera mechanism
    # fills whatever the policy declares and the dataset does not have.
    if not (set(p_vis) >= set(d_vis) or set(d_vis) >= set(p_vis)):
        print("  FAIL  neither is a subset of the other -- training will refuse to start")
        print(f"        missing: {sorted(set(p_vis) - set(d_vis))}")
        print(f"        extra  : {sorted(set(d_vis) - set(p_vis))}")
        ok = False
    else:
        print("  ok    subset check passes")

    a_pol = (pout.get("action") or {}).get("shape")
    a_ds = (dfeat.get("action") or {}).get("shape")
    s_pol = (pin.get("observation.state") or {}).get("shape")
    s_ds = (dfeat.get("observation.state") or {}).get("shape")
    print()
    print(f"action  policy {a_pol}  dataset {a_ds}" + ("   ok" if a_pol == a_ds else "   FAIL"))
    print(f"state   policy {s_pol}  dataset {s_ds}" + ("   ok" if s_pol == s_ds else "   FAIL"))
    ok = ok and a_pol == a_ds and s_pol == s_ds

    print()
    print("PREFLIGHT PASS" if ok else "PREFLIGHT FAIL -- fix this before spending the hour")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
