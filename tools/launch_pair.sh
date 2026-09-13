#!/usr/bin/env bash
# Launch train_pair.sh detached, from a snapshot of it.
#
# Two ways a long run has been lost on this machine, both fixed here.
#
# setsid + nohup: a run was killed at step 69506 with no traceback because the
# shell that launched it went away. Process parentage, not any fault in training.
#
# The copy: bash reads a script incrementally as it executes, so editing the file
# a run is executing changes what that run does next. On 2026-09-08 a one-line
# fix to the retry logic made a four-hour pipeline re-execute from the top
# mid-flight and abandon the model it was training. Copying to a path nobody
# edits makes the running job immune to whatever happens in the repo.
set -euo pipefail

BASE=/home/padiac/lerobot-train
SRC=/mnt/e/Repo/so101-build/tools/train_pair.sh

if pgrep -f "bin/lerobot-train" > /dev/null; then
    echo "a training process is already running -- not starting another"
    pgrep -af "bin/lerobot-train" | head -2
    exit 1
fi

mkdir -p "$BASE/logs" "$BASE/run"
SNAP=$BASE/run/pair_$(date +%Y%m%d_%H%M%S).sh
cp "$SRC" "$SNAP"

setsid nohup bash "$SNAP" > "$BASE/logs/pair_launch.log" 2>&1 < /dev/null &
sleep 5
echo "launched from $SNAP"
pgrep -af "run/pair_.*\.sh" | head -2
