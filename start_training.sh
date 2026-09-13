#!/usr/bin/env bash
# Start (or restart) a detached training run inside WSL.
#
# Launching lerobot-train as a child of `wsl.exe` ties its life to whatever
# started that client. On 2026-09-01 a run was killed at step 69506 with no
# traceback and no error simply because the shell that launched it went away --
# 9500 steps lost to process parentage, not to any fault in the training.
#
# setsid + nohup detaches it from the session entirely, so the run survives the
# terminal, the agent, and anything else that launched it. It is wrapped in
# train_resume.sh so a genuine crash is also retried automatically.
#
#   bash start_training.sh [output_dir]
set -euo pipefail

BASE=/home/padiac/lerobot-train
OUT=${1:-$(ls -dt "$BASE"/outputs/*/ 2>/dev/null | head -1)}
OUT=${OUT%/}
HERE=$(dirname "$(readlink -f "$0")")

if pgrep -f "bin/lerobot-train" > /dev/null; then
    echo "a training process is already running (pid $(pgrep -f "bin/lerobot-train" | head -1)) -- not starting another"
    exit 1
fi

mkdir -p "$BASE/logs"
LOG="$BASE/logs/resume_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$LOG" "$BASE/logs/current.log"

setsid nohup bash "$HERE/train_resume.sh" "$OUT" > "$LOG" 2>&1 < /dev/null &
sleep 3
echo "started detached"
echo "  run  $(basename "$OUT")"
echo "  log  $LOG  (also $BASE/logs/current.log)"
pgrep -af "train_resume.sh|bin/lerobot-train" | head -3
