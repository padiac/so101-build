#!/usr/bin/env bash
# SmolVLA on so101_emptybowl, queued behind whatever is training now.
#
# Runs second on purpose. The diffusion model is the one expected to work here,
# so it gets the GPU first and the guaranteed slot; this one takes what is left
# before the morning. It waits for the current training to exit rather than
# fighting it for memory.
#
# One fixed instruction, so the language channel has nothing to do -- this is
# not a language test. It is the same-task comparison that every other dataset
# here has: on so101_v3, ACT and diffusion both beat SmolVLA, and the question
# is whether that still holds when the task is six grasps in clutter instead of
# one in the open.
#
# Data is whatever train_emptybowl.sh prepared: 41 episodes, the stopped-early
# one dropped, the idle head trimmed. This script never rebuilds it.
#
#   bash tools/train_emptybowl_smolvla.sh
#   STEPS=70000 DEADLINE=08:30 bash tools/train_emptybowl_smolvla.sh
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
SRC=so101_emptybowl
STEPS=${STEPS:-80000}
BATCH=${BATCH:-8}
SAVE_EVERY=${SAVE_EVERY:-5000}
MAX_TRIES=${MAX_TRIES:-3}
FREE_MB=${FREE_MB:-4000}
# Measured 2026-09-25: SmolVLA runs at 4.68 step/s at batch 8 on this card.
RATE=${RATE:-4.68}
DEADLINE=${DEADLINE:-}

RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
LOG=$BASE/logs/emptybowl_smolvla_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== smolvla on $SRC queued: $STEPS steps, batch $BATCH ==="
[ -d "$BASE/data/${SRC}_trim" ] || { say "!!! $BASE/data/${SRC}_trim missing; run train_emptybowl.sh first"; exit 2; }

# Wait for the whole first pipeline to exit, not just its training process.
# Keying on "is diffusion training right now" was wrong and cost a crash: that
# pipeline spends its first half hour copying and re-encoding the dataset, so
# the condition was true before training had even started, and this script read
# a half-written parquet. The pipeline script's own process is the thing to
# wait on, and matching the run/ snapshot name also survives its retries.
waited=0
while pgrep -f "run/emptybowl_[0-9]" | grep -qv "^$$\$"; do
    [ "$waited" -eq 0 ] && say "the diffusion pipeline is still running; waiting"
    waited=$((waited + 1))
    [ "$waited" -gt 960 ] && { say "still running after 8h; giving up on the queue"; exit 3; }
    sleep 30
done
# And make sure nothing else grabbed the GPU in the gap. Match the binary under
# .venv/bin, not the bare name: the project lives in /home/padiac/lerobot-train,
# so "lerobot-train" appears in this script's own command line and the check
# matched itself. That cost three hours of an idle GPU on 2026-10-10 -- the
# diffusion run finished at 04:35 and this one was still waiting at 07:48.
while pgrep -f "bin/lerobot-train" > /dev/null; do sleep 30; done
say "the GPU is free"

# Trim the plan to whatever time is left, rather than overrunning the morning.
if [ -n "$DEADLINE" ]; then
    target=$(date -d "today $DEADLINE" +%s 2>/dev/null)
    now=$(date +%s)
    [ "$target" -le "$now" ] && target=$(date -d "tomorrow $DEADLINE" +%s)
    left=$(( target - now ))
    fits=$(awk -v l="$left" -v r="$RATE" 'BEGIN{printf "%d", l * r}')
    # Round down to a whole checkpoint so the published model sits on one.
    fits=$(( fits / SAVE_EVERY * SAVE_EVERY ))
    say "$(( left / 60 )) min until $DEADLINE; at $RATE step/s that is $fits steps"
    if [ "$fits" -lt "$STEPS" ]; then
        if [ "$fits" -lt "$SAVE_EVERY" ]; then
            say "not enough time for even one checkpoint -- not starting"
            exit 0
        fi
        say "cutting $STEPS to $fits to land before $DEADLINE"
        STEPS=$fits
    fi
fi

waited=0
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt "$FREE_MB" ] && break
    [ "$waited" -eq 0 ] && say "GPU busy (${used} MiB); waiting"
    waited=$((waited + 1))
    [ "$waited" -gt 120 ] && { say "GPU still busy after 1h; starting anyway"; break; }
    sleep 30
done

last_step() {
    local d
    d=$(ls -d "$1"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1) || return 1
    [ -n "$d" ] && basename "$d" || echo 0
}

out=$BASE/outputs/smolvla_${SRC}_$(date +%Y%m%d_%H%M)
mlog=$BASE/logs/smolvla_${SRC}_$(date +%Y%m%d_%H%M).log
rc=1
for (( try = 1; try <= MAX_TRIES; try++ )); do
    prev=$(last_step "$out" 2>/dev/null || echo 0)
    if [ ! -d "$out/checkpoints/last" ]; then
        say "=== train attempt $try ($STEPS steps, batch $BATCH) ==="
        "$V/lerobot-train" \
            --policy.path=lerobot/smolvla_base \
            --dataset.repo_id="local/${SRC}_trim" \
            --dataset.root="$BASE/data/${SRC}_trim" \
            --rename_map="$RENAME" \
            --policy.device=cuda --policy.push_to_hub=false \
            --policy.scheduler_decay_steps="$STEPS" \
            --output_dir="$out" --job_name="smolvla_${SRC}" \
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

name=smolvla_emptybowl_$(( STEPS / 1000 ))k
say "publishing -> policies/$name"
rm -rf "${WIN:?}/policies/$name"
cp -r "$out/checkpoints/last/pretrained_model" "$WIN/policies/$name"
say "done: $(grep -aoE 'step:[0-9]+K [^|]*loss:[0-9.]+' "$mlog" | tail -1)"
