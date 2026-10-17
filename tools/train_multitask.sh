#!/usr/bin/env bash
# Diffusion on the multi-task dataset: emptybowl + color6 + v3, each frame
# tagged with a one-hot task id (tools/build_multitask.py). Deployed with the
# emptybowl id, the other two tasks contribute their grasps without teaching it
# to stop after one cube or to put cubes back in.
#
# emptybowl is 51% of the frames (214715 of 423144): its episodes are long,
# color6's are one cube each. 150k steps at batch 8 is about 2.8 passes.
#
#   bash tools/train_multitask.sh
#   START_AT=02:00 bash tools/train_multitask.sh
#   RESUME_OUT=<outputs dir> bash tools/train_multitask.sh
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
PREP=so101_multitask_trim
STEPS=${STEPS:-150000}
BATCH=${BATCH:-8}
SAVE_EVERY=${SAVE_EVERY:-5000}
MAX_TRIES=${MAX_TRIES:-3}
FREE_MB=${FREE_MB:-4000}
START_AT=${START_AT:-}
RESIZE='[240,320]'
SRC=multitask

LOG=$BASE/logs/multitask_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if [ -n "$START_AT" ]; then
    target=$(date -d "today $START_AT" +%s 2>/dev/null)
    now=$(date +%s)
    [ "$target" -le "$now" ] && target=$(date -d "tomorrow $START_AT" +%s)
    say "waiting for $START_AT on $(date -d "@$target" '+%m-%d') (in $(( (target - now) / 60 )) min)"
    while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done
fi

count_episodes() {
    "$V/python" -c "import json,sys,pathlib;p=pathlib.Path(sys.argv[1])/'meta'/'info.json';print(json.loads(p.read_text())['total_episodes'] if p.exists() else -1)" "$1"
}
[ -f "$BASE/data/$PREP/task_vector.json" ] || {
    say "building $PREP"
    "$V/python" "$WIN/tools/build_multitask.py" 2>&1 | grep -v "it/s" | tail -6 | tee -a "$LOG"
}
KEPT=$(count_episodes "$BASE/data/$PREP")
[ "$KEPT" -gt 0 ] || { say "!!! $PREP missing"; exit 3; }
say "=== multitask: $KEPT episodes, $STEPS steps ==="

waited=0
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt "$FREE_MB" ] && break
    [ "$waited" -eq 0 ] && say "GPU busy (${used} MiB); waiting"
    waited=$((waited + 1))
    [ "$waited" -gt 240 ] && { say "GPU still busy after 2h; starting anyway"; break; }
    sleep 30
done

last_step() {
    local d
    d=$(ls -d "$1"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1) || return 1
    [ -n "$d" ] && basename "$d" || echo 0
}

# A full checkpoint is ~1 GB of weights plus ~2 GB of optimizer state, and this
# run writes one every SAVE_EVERY steps. On 2026-10-15 the WSL disk filled at
# step 20000 -- 661 GB of old runs had never been pruned -- and the run died
# writing its checkpoint, twice. Refuse up front instead of failing an hour in.
NEED_GB=${NEED_GB:-$(( (STEPS / SAVE_EVERY + 1) * 3 + 10 ))}
free_gb=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt "$NEED_GB" ]; then
    say "!!! only ${free_gb} GB free under $BASE, this run needs about ${NEED_GB} GB"
    say "    optimizer state of finished runs is safe to delete:"
    say "    rm -rf $BASE/outputs/*/checkpoints/*/training_state   (finished runs only)"
    exit 4
fi
say "disk: ${free_gb} GB free, about ${NEED_GB} GB needed"

# RESUME_OUT continues a run that died outright -- WSL shut down, the machine
# rebooted -- rather than starting a fresh directory and losing its steps. The
# retry loop below already resumes whenever checkpoints/last exists.
out=${RESUME_OUT:-$BASE/outputs/diffusion_${SRC}_$(date +%Y%m%d_%H%M)}
mlog=$BASE/logs/$(basename "$out").log
say "output: $out"
rc=1
for (( try = 1; try <= MAX_TRIES; try++ )); do
    prev=$(last_step "$out" 2>/dev/null || echo 0)
    if [ ! -d "$out/checkpoints/last" ]; then
        say "=== train attempt $try ($STEPS steps, batch $BATCH) ==="
        "$V/lerobot-train" \
            --policy.type=diffusion \
            --dataset.repo_id="local/$PREP" \
            --dataset.root="$BASE/data/$PREP" \
            --policy.device=cuda --policy.push_to_hub=false \
            --policy.resize_shape="$RESIZE" --policy.crop_ratio=0.95 \
            --output_dir="$out" --job_name="diffusion_${SRC}" \
            --steps="$STEPS" --save_freq="$SAVE_EVERY" --log_freq=500 \
            --batch_size="$BATCH" --num_workers=4 --prefetch_factor=2 \
            --wandb.enable=false 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
    else
        say "=== resume attempt $try (from $(last_step "$out")) ==="
        "$V/lerobot-train" --resume=true \
            --config_path="$out/checkpoints/last/pretrained_model" 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
    fi
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && break
    now=$(last_step "$out" 2>/dev/null || echo 0)
    say "exited $rc at step $now (was $prev)"
    if [ "$try" -gt 1 ] && [ "$now" != "0" ] && [ "$now" = "$prev" ]; then
        say "no progress since the last attempt -- a real fault"
        break
    fi
    say "retrying in 120s"
    sleep 120
done

[ "$rc" -eq 0 ] || { say "!!! FAILED"; exit 1; }

# The run is finished, so nothing will ever resume it: drop the optimizer state
# of every checkpoint. The weights stay -- intermediate checkpoints are what the
# training-curve measurements in MODELS.md were taken from.
pruned=$(du -smc "$out"/checkpoints/[0-9]*/training_state 2>/dev/null | tail -1 | cut -f1)
rm -rf "$out"/checkpoints/[0-9]*/training_state
say "pruned ${pruned:-0} MB of optimizer state; weights kept"

name=diffusion_multitask${KEPT}_$(( STEPS / 1000 ))k_trim
say "publishing -> policies/$name"
rm -rf "${WIN:?}/policies/$name"
cp -r "$out/checkpoints/last/pretrained_model" "$WIN/policies/$name"
cp "$BASE/data/$PREP/task_vector.json" "$WIN/policies/$name/"
say "done: $(grep -aoE 'step:[0-9]+K [^|]*loss:[0-9.]+' "$mlog" | tail -1)"
