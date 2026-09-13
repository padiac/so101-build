#!/usr/bin/env bash
# Train the colour-agnostic model on all 150 out-of-bowl demonstrations.
#
# so101_color6 is 150 demonstrations of one motion split five ways by colour, so
# each instruction has only 30. The model trained on it does not reach for any
# cube, which is a failure of the motion rather than of colour discrimination.
# This puts all 150 behind a single instruction and asks whether the motion is
# learnable at all. Any cube grasped counts.
#
# Everything except the dataset matches the other two 2026-09-08 models -- same
# base checkpoint, same rename map, batch 8, 20000 steps -- so a difference is a
# difference in the data.
#
#   START_AT=02:00 bash tools/train_c6pool.sh
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
DS=so101_c6pool
PUBLISH=smolvla_c6pool_20k
STEPS=${STEPS:-20000}
BATCH=${BATCH:-8}
MAX_TRIES=${MAX_TRIES:-4}
START_AT=${START_AT:-}

RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

LOG=$BASE/logs/c6pool_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if [ -n "$START_AT" ]; then
    # Sleep to the wall clock rather than for a duration: the delay is armed at
    # one time and read by a person who cares about the other.
    target=$(date -d "today $START_AT" +%s 2>/dev/null)
    now=$(date +%s)
    [ "$target" -le "$now" ] && target=$(date -d "tomorrow $START_AT" +%s)
    say "armed; starting at $(date -d "@$target" '+%H:%M') "\
"(in $(( (target - now) / 60 )) min)"
    while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done
fi

[ -d "$BASE/data/${DS}_trim" ] || { say "!!! missing ${DS}_trim"; exit 3; }

# Wait for the GPU. A deployment session on the Windows side holds about 3 GB of
# the 10, and this machine already refuses to train at batch 16 with room to
# spare, so starting into an occupied card is how a whole night gets wasted. Idle
# desktop sits near 1.6 GB.
FREE_MB=${FREE_MB:-3000}
waited=0
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt "$FREE_MB" ] && break
    [ "$waited" -eq 0 ] && say "GPU busy (${used} MiB in use); waiting for it to free up"
    waited=$((waited + 1))
    if [ "$waited" -gt 360 ]; then          # 3 hours at 30s
        say "GPU still busy after 3h (${used} MiB); starting anyway"
        break
    fi
    sleep 30
done
[ "$waited" -gt 0 ] && say "GPU free after $((waited / 2)) min, starting"

say "=== preflight ==="
"$V/python" "$WIN/preflight_train.py" "$BASE/data/${DS}_trim" \
    --policy "$WIN/survey/models/smolvla_base" --rename "$RENAME" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "!!! preflight failed"; exit 3; }

last_step() {
    local d
    d=$(ls -d "$1"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1) || return 1
    [ -n "$d" ] && basename "$d" || echo 0
}

OUT=$BASE/outputs/smolvla_${DS}_$(date +%Y%m%d_%H%M)
rc=1
for (( try = 1; try <= MAX_TRIES; try++ )); do
    prev=$(last_step "$OUT" 2>/dev/null || echo 0)
    if [ ! -d "$OUT/checkpoints/last" ]; then
        say "=== train $DS  attempt $try  ($STEPS steps, batch $BATCH) ==="
        "$V/lerobot-train" \
            --policy.path=lerobot/smolvla_base \
            --dataset.repo_id="local/${DS}_trim" \
            --dataset.root="$BASE/data/${DS}_trim" \
            --rename_map="$RENAME" \
            --policy.device=cuda --policy.push_to_hub=false \
            --output_dir="$OUT" --job_name="smolvla_${DS}" \
            --steps="$STEPS" --save_freq=2000 --log_freq=200 \
            --batch_size="$BATCH" --num_workers=4 --prefetch_factor=2 \
            --wandb.enable=false 2>&1 | tee -a "$LOG"
    else
        say "=== resume $DS  attempt $try  (from $(last_step "$OUT")) ==="
        "$V/lerobot-train" --resume=true \
            --config_path="$OUT/checkpoints/last/pretrained_model" 2>&1 | tee -a "$LOG"
    fi
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && break

    now=$(last_step "$OUT" 2>/dev/null || echo 0)
    say "$DS exited $rc at step $now (was $prev)"
    # Only a repeat failure at the SAME checkpoint is a real fault. Written as a
    # bare equality this also fired on the first failure, before any checkpoint
    # existed, and abandoned a model over one transient CUDA error.
    if [ "$try" -gt 1 ] && [ "$now" != "0" ] && [ "$now" = "$prev" ]; then
        say "no progress since the last attempt -- a real fault, not a blip"
        break
    fi
    say "retrying in 120s"
    sleep 120
done

if [ "$rc" -ne 0 ]; then
    say "!!! $DS FAILED, see $LOG"
    exit 1
fi

say "publishing -> policies/$PUBLISH"
rm -rf "${WIN:?}/policies/$PUBLISH"
cp -r "$OUT/checkpoints/last/pretrained_model" "$WIN/policies/$PUBLISH" 2>&1 | tee -a "$LOG"
say "=== $DS done ==="
grep -oE "step:20K [^|]*loss:[0-9.]+" "$LOG" | tail -1 | tee -a "$LOG"
