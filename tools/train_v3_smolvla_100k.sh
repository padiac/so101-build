#!/usr/bin/env bash
# SmolVLA on so101_v3 for 100k steps at batch 16, with an automatic fall back to 8.
#
# What this is for: the 20k and 40k models on this data are both far rougher than
# ACT and diffusion, and 20k -> 40k moved roughness 1.44 -> 1.03. Whether that
# keeps going, and whether the batch size was holding it back, are the two
# untested knobs that cost nothing but a night.
#
# Batch 16 was measured on 2026-09-25 at 2.81 step/s and 4.61 GB, so 100k steps is
# about ten hours. Batch 16 also failed three times on 2026-09-08 in a driver
# memory bug that did not reproduce -- hence the fall back rather than a choice.
# A fall back restarts from zero rather than resuming, because a resumed run
# keeps the batch size recorded in its own config and the point here is the batch.
#
# No validation split, deliberately. smolvla_v3only_20k and _40k were both
# trained on all 130 episodes, and holding 10% out now would change two things at
# once. Checkpoints every 5000 steps instead, so the curve can be measured
# offline afterwards the way the ACT curve was.
#
#   bash tools/train_v3_smolvla_100k.sh
#   START_AT=02:50 bash tools/train_v3_smolvla_100k.sh
#   BATCH=8 STEPS=40000 bash tools/train_v3_smolvla_100k.sh
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
DS=so101_v3
STEPS=${STEPS:-100000}
BATCH=${BATCH:-16}
FALLBACK=${FALLBACK:-8}
SAVE_EVERY=${SAVE_EVERY:-5000}
FREE_MB=${FREE_MB:-3500}
MAX_TRIES=${MAX_TRIES:-3}
START_AT=${START_AT:-}

RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
LOG=$BASE/logs/v3smol100k_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== smolvla on $DS: $STEPS steps, batch $BATCH (fall back $FALLBACK) ==="
[ -d "$BASE/data/${DS}_trim" ] || { say "!!! $BASE/data/${DS}_trim missing"; exit 2; }

if [ -n "$START_AT" ]; then
    # The next occurrence of that clock time, capped: rolling to tomorrow without
    # a cap once put a run 22 hours out, and refusing to roll once started a
    # 00:30 job at 21:32 the evening before.
    MAX_WAIT_H=${MAX_WAIT_H:-12}
    target=$(date -d "today $START_AT" +%s 2>/dev/null)
    now=$(date +%s)
    [ "$target" -le "$now" ] && target=$(date -d "tomorrow $START_AT" +%s)
    wait_s=$(( target - now ))
    if [ "$wait_s" -gt $(( MAX_WAIT_H * 3600 )) ]; then
        say "$START_AT is $(( wait_s / 3600 ))h away, past the ${MAX_WAIT_H}h cap -- starting now"
    else
        say "waiting for $START_AT on $(date -d "@$target" '+%m-%d') (in $(( wait_s / 60 )) min)"
        while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done
    fi
fi

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

run_at_batch() {       # $1 = batch. 0 = finished, 1 = out of memory, 2 = other fault
    local b=$1 out mlog try prev now rc
    out=$BASE/outputs/smolvla_${DS}_b${b}_$(date +%Y%m%d_%H%M)
    mlog=$BASE/logs/smolvla_${DS}_b${b}_$(date +%Y%m%d_%H%M).log
    PUBLISH_FROM=$out
    for (( try = 1; try <= MAX_TRIES; try++ )); do
        prev=$(last_step "$out" 2>/dev/null || echo 0)
        if [ ! -d "$out/checkpoints/last" ]; then
            say "=== batch $b, attempt $try ==="
            "$V/lerobot-train" \
                --policy.path=lerobot/smolvla_base \
                --dataset.repo_id="local/${DS}_trim" \
                --dataset.root="$BASE/data/${DS}_trim" \
                --rename_map="$RENAME" \
                --policy.device=cuda --policy.push_to_hub=false \
                --policy.scheduler_decay_steps="$STEPS" \
                --output_dir="$out" --job_name="smolvla_${DS}_b${b}" \
                --steps="$STEPS" --save_freq="$SAVE_EVERY" --log_freq=500 \
                --batch_size="$b" --num_workers=4 --prefetch_factor=2 \
                --wandb.enable=false 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
        else
            say "=== batch $b, resume attempt $try (from $(last_step "$out")) ==="
            "$V/lerobot-train" --resume=true \
                --config_path="$out/checkpoints/last/pretrained_model" 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
        fi
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] && return 0
        now=$(last_step "$out" 2>/dev/null || echo 0)
        say "batch $b exited $rc at step $now (was $prev)"
        # Out of memory before a single checkpoint means this batch does not fit
        # on this card today; retrying it would only spend the night failing.
        if grep -aq "OutOfMemoryError\|out of memory" "$mlog" && [ "$now" = "0" ]; then
            say "out of memory at batch $b with nothing saved"
            return 1
        fi
        if [ "$try" -gt 1 ] && [ "$now" != "0" ] && [ "$now" = "$prev" ]; then
            say "no progress since the last attempt -- a real fault"
            return 2
        fi
        say "retrying in 120s"
        sleep 120
    done
    return 2
}

PUBLISH_FROM=""
run_at_batch "$BATCH"; rc=$?
if [ "$rc" -eq 1 ] && [ "$FALLBACK" -gt 0 ] && [ "$FALLBACK" -ne "$BATCH" ]; then
    say "falling back to batch $FALLBACK, starting over"
    BATCH=$FALLBACK
    run_at_batch "$BATCH"; rc=$?
fi
if [ "$rc" -ne 0 ]; then
    say "!!! FAILED (rc=$rc)"
    exit 1
fi

name=smolvla_v3only_$(( STEPS / 1000 ))k_b${BATCH}
say "publishing -> policies/$name"
rm -rf "${WIN:?}/policies/$name"
cp -r "$PUBLISH_FROM/checkpoints/last/pretrained_model" "$WIN/policies/$name"
say "done at batch $BATCH: $(grep -aoE 'step:[0-9]+K [^|]*loss:[0-9.]+' "$BASE/logs/smolvla_${DS}_b${BATCH}"*.log | tail -1)"
