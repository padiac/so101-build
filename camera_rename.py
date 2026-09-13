r"""Work out the --rename_map a policy needs, from the policy itself.

Camera key names are a per-checkpoint convention, not a standard: ACT trained here
expects observation.images.top/wrist, while lerobot/smolvla_base expects
camera1/camera2/camera3. Deploying the wrong pairing fails at startup with a
feature mismatch -- it cost a training run on 2026-09-05 and a rollout on 09-07,
the same mistake on both sides of the pipeline.

Rather than hard-code a map per policy, read what the checkpoint declares and pair
it positionally with the cameras this robot actually has. Prints a JSON map, or
nothing at all when the names already agree.

    .\.venv-win\Scripts\python.exe camera_rename.py policies\smolvla_mix1_100k top wrist
"""

import json
import sys
from pathlib import Path

PREFIX = "observation.images."


def rename_map(policy_dir: Path, robot_cams: list[str]) -> dict[str, str]:
    cfg = json.loads((policy_dir / "config.json").read_text(encoding="utf-8"))
    want = [k for k, v in (cfg.get("input_features") or {}).items() if v.get("type") == "VISUAL"]
    want.sort()
    have = [PREFIX + c for c in robot_cams]

    if set(have) <= set(want) or set(want) <= set(have):
        return {}          # already compatible; lerobot pads or ignores the rest

    # Positional pairing. The order the robot lists its cameras is the order they
    # were recorded in, and the policy's camera1/2/3 follow that same order, so
    # this reproduces the mapping training used.
    return {h: w for h, w in zip(have, want)}


def main() -> int:
    argv = sys.argv[1:]
    ps = "--ps" in argv
    argv = [a for a in argv if a != "--ps"]      # strip flags before reading camera names
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    m = rename_map(Path(argv[0]), argv[1:])
    if not m:
        return 0
    out = json.dumps(m, separators=(",", ":"))
    # PowerShell strips bare double quotes when it hands an argument to a native
    # command, so `--rename_map={"a":"b"}` arrives as `{a:b}` -- not valid JSON,
    # silently parsed as empty, and the rename never happens while the startup
    # check still passes because the raw string was non-empty. Escaping the inner
    # quotes is what survives the trip.
    if ps:
        out = out.replace(chr(34), chr(92) + chr(34))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
