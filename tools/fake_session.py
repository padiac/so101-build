r"""Stand in for a live rollout, so the panel can be checked without a robot.

It imitates the two things the panel reads: the session log, where lerobot writes
"control loop started" once the policy is driving, and task.status, which
live_task.py rewrites on every state change. The timings are the measured ones --
about eighty seconds of setup, three seconds of homing before each instruction --
compressed so a test finishes in seconds.

Started automatically by `panel.py --stub`; not useful on its own.
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
# The panel tells us which file to use, exactly as it tells the real rollout via
# robot.ps1. A test can then be pointed away from the live one.
TASK = Path(os.environ.get("LIVE_TASK_FILE", HERE / "task.txt"))
STATUS = TASK.with_suffix(".status")

BOOT_S = float(os.environ.get("FAKE_BOOT", 2.0))     # stands in for ~80s of setup
HOME_S = float(os.environ.get("FAKE_HOME", 1.5))     # stands in for ~3s of homing


def status(mode, task=""):
    STATUS.write_text(json.dumps({"mode": mode, "task": task}), encoding="utf-8")


def read_task():
    try:
        return TASK.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def main():
    status("idle")
    time.sleep(BOOT_S)
    # The panel waits for this line in the log before it enables anything.
    print("Base strategy control loop started", flush=True)

    current = ""
    while True:
        want = read_task()
        if want != current:
            # live_task returns the arm to the training start pose between
            # instructions, including on the way to standby.
            status("homing", want)
            time.sleep(HOME_S)
            current = want
            status("running" if current else "idle", current)
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
