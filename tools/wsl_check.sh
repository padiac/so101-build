#!/usr/bin/env bash
# What is on the training side right now: datasets, GPU, disk, running jobs.
#
# Quoting a for-loop through `wsl.exe -- bash -lc '...'` from Git Bash loses the
# loop variable, so anything with a variable in it lives in a file and is run as
# a file.
BASE=/home/padiac/lerobot-train

echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "=== training running ==="
pgrep -af "bin/lerobot-train" || echo "none"

echo "=== disk ==="
df -h "$BASE" | tail -1

echo "=== datasets in ext4 ==="
for d in "$BASE"/data/*/; do
    n=$(basename "$d")
    if [ -f "$d/meta/info.json" ]; then
        printf "  %-22s " "$n"
        python3 - "$d/meta/info.json" <<'PY'
import json, sys
i = json.load(open(sys.argv[1]))
print("%4d eps  %7d frames" % (i["total_episodes"], i["total_frames"]))
PY
    else
        printf "  %-22s no meta/info.json\n" "$n"
    fi
done
