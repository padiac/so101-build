#!/usr/bin/env bash
# Arm the colour-6 pipeline detached, so it survives whatever launched it.
set -euo pipefail
B=/home/padiac/lerobot-train
mkdir -p "$B/logs"
if pgrep -f "bin/lerobot-train" > /dev/null || pgrep -f pipeline_color6.sh > /dev/null; then
    echo "already armed or running:"
    pgrep -af "pipeline_color6.sh|bin/lerobot-train" | head -3
    exit 1
fi
LOG=$B/logs/color6_armed_$(date +%Y%m%d_%H%M%S).log
setsid nohup env DELAY="${DELAY:-7200}" STEPS="${STEPS:-20000}" BATCH="${BATCH:-16}" \
    bash /mnt/e/Repo/so101-build/pipeline_color6.sh > "$LOG" 2>&1 < /dev/null &
sleep 2
echo "armed. log: $LOG"
echo "starts at: $(date -d "+${DELAY:-7200} seconds" '+%H:%M')"
pgrep -af pipeline_color6.sh | head -2
