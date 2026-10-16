#!/usr/bin/env bash
# Tonight's queue: diffusion, then SmolVLA, on so101_emptybowl, detached.
#
# One wrapper runs them back to back, so there is no queue to poll. The second
# runs even if the first fails -- they are independent models on the same
# prepared data, and a dead diffusion run should not cost the SmolVLA one.
#
# Both scripts are snapshotted first, for the reason in launch_pair.sh: bash
# reads a script as it executes it, and an edit in the repo mid-run changes
# what the running job does next. setsid + nohup so the job outlives the shell.
#
#   START_AT=02:00 bash tools/launch_night.sh
set -euo pipefail

BASE=/home/padiac/lerobot-train
TOOLS=/mnt/e/Repo/so101-build/tools

if pgrep -f "bin/lerobot-train|$BASE/run/" > /dev/null; then
    echo "a training job is already running -- not starting another"
    pgrep -af "bin/lerobot-train|$BASE/run/" | head -3
    exit 1
fi

SNAP=$BASE/run/night_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SNAP" "$BASE/logs"
cp "$TOOLS/train_emptybowl.sh" "$SNAP/diffusion.sh"
cp "$TOOLS/train_emptybowl_smolvla.sh" "$SNAP/smolvla.sh"
cat > "$SNAP/queue.sh" <<Q
#!/usr/bin/env bash
START_AT=${START_AT:-} bash "$SNAP/diffusion.sh"
echo "diffusion exited \$?"
bash "$SNAP/smolvla.sh"
echo "smolvla exited \$?"
Q

setsid nohup bash "$SNAP/queue.sh" > "$SNAP/queue.log" 2>&1 < /dev/null &
sleep 5
echo "launched from $SNAP"
pgrep -af "$SNAP" | head -3
