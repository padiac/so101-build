#!/usr/bin/env bash
# Train two single-task SmolVLA models back to back, unattended.
#
# The existing SmolVLA checkpoints were both trained on the union of the two
# recorded sets (so101_mix1). That answers "can one policy hold both tasks" but
# it cannot answer "does either task work at all", because a failure in the mixed
# model has two possible causes and no way to tell them apart. These two are the
# controls: each sees exactly one task.
#
#   A  so101_v3      130 episodes, put the yellow block INTO the bowl
#   B  so101_color6  150 episodes, take a named cube OUT of the bowl
#
# The two differ from each other ONLY in the dataset -- same base checkpoint,
# same rename map, same batch, same step count -- so a difference between them is
# a difference in the data. See the BATCH note below for how they differ from the
# mixed runs, which is not by choice.
#
# One GPU, so they run in sequence: about 75 minutes each at 4.4 step/s.
#
#   bash tools/train_pair.sh            # run it
#   STEPS=20000 bash tools/train_pair.sh
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
STEPS=${STEPS:-20000}
# Batch 8, not the 16 the mix1 runs used. On 2026-09-08 batch 16 died three times
# in the first forward pass, in SmolVLA's eager attention, asking for 48 MiB with
# 4.84 GiB free while reporting its own usage as 17179869184 GiB -- which is 2^64
# bytes exactly, an unsigned subtraction that wrapped because the driver returned
# a free value larger than total. torch 2.11.0+cu130, driver 581.95, through WSL.
# Batch 8 steps cleanly at 4.4 step/s using 2.96 GB. Nothing else is changed, so
# the two models here stay exactly comparable with each other; they are NOT
# strictly comparable with smolvla_mix1_* , which used batch 16.
BATCH=${BATCH:-8}
MAX_TRIES=${MAX_TRIES:-3}

# smolvla_base names its cameras camera1/2/3; these datasets name them top/wrist.
# Training died in its first second over this once, after ninety minutes of data
# preparation, which is why preflight now runs before anything expensive.
RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

STAMP=$(date +%Y%m%d_%H%M)
LOG=$BASE/logs/pair_$STAMP.log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Read the step of the newest checkpoint, so a retry that made no progress can be
# told from one that did. Retrying a run that dies at the same step twice just
# burns the day and produces identical tracebacks.
last_step() {
    local d
    d=$(ls -d "$1"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1) || return 1
    [ -n "$d" ] && basename "$d" || echo 0
}

trim_if_needed() {   # $1 = dataset name
    if [ -d "$BASE/data/$1_trim" ]; then
        say "$1_trim already exists, keeping it"
        return 0
    fi
    say "trimming $1 -> $1_trim"
    cp -f "$WIN/trim_dataset.py" "$BASE/trim_dataset.py"
    "$V/python" "$BASE/trim_dataset.py" --src "$BASE/data/$1" \
        --dst "$BASE/data/$1_trim" --force 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

train_one() {        # $1 = dataset name, $2 = name to publish under
    local ds=$1 out_name=$2 out try prev now rc
    out=$BASE/outputs/smolvla_${ds}_$STAMP

    for (( try = 1; try <= MAX_TRIES; try++ )); do
        prev=$(last_step "$out" 2>/dev/null || echo 0)
        if [ "$try" -eq 1 ] || [ ! -d "$out/checkpoints/last" ]; then
            say "=== train $ds  attempt $try  ($STEPS steps, batch $BATCH) ==="
            "$V/lerobot-train" \
                --policy.path=lerobot/smolvla_base \
                --dataset.repo_id="local/${ds}_trim" \
                --dataset.root="$BASE/data/${ds}_trim" \
                --rename_map="$RENAME" \
                --policy.device=cuda \
                --policy.push_to_hub=false \
                --output_dir="$out" \
                --job_name="smolvla_${ds}_$STAMP" \
                --steps="$STEPS" \
                --save_freq=2000 --log_freq=200 \
                --batch_size="$BATCH" \
                --num_workers=4 --prefetch_factor=2 \
                --wandb.enable=false 2>&1 | tee -a "$LOG"
        else
            say "=== resume $ds  attempt $try  (from $(last_step "$out")) ==="
            "$V/lerobot-train" --resume=true \
                --config_path="$out/checkpoints/last/pretrained_model" 2>&1 | tee -a "$LOG"
        fi
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] && break

        now=$(last_step "$out" 2>/dev/null || echo 0)
        say "$ds exited $rc at step $now (was $prev)"
        # The guard only means something once a checkpoint exists and there has
        # been a previous attempt to compare against. Written as a bare
        # "now = prev" it also fired on the FIRST failure before any checkpoint
        # was written, where both are zero -- so a transient CUDA allocation
        # failure sixty seconds in was treated as a permanent fault and the model
        # was abandoned without a single retry. That is the one case where a
        # retry is most likely to work, not least.
        if [ "$try" -gt 1 ] && [ "$now" != "0" ] && [ "$now" = "$prev" ]; then
            say "no progress since the last attempt -- this is a real fault, not a blip"
            break
        fi
        say "retrying in 60s"
        sleep 60
    done

    if [ "$rc" -ne 0 ]; then
        say "!!! $ds FAILED after $try attempts, see $LOG"
        return 1
    fi

    # Publish to the Windows side straight away. Otherwise a finished model sits
    # in ext4 until someone remembers to fetch it, and the person who wanted it
    # has been out all day.
    local dst=$WIN/policies/$out_name
    say "publishing -> policies/$out_name"
    rm -rf "$dst"
    cp -r "$out/checkpoints/last/pretrained_model" "$dst" 2>&1 | tee -a "$LOG"
    say "$ds done, final loss line:"
    grep -o "loss:[0-9.]*" "$LOG" | tail -1 | tee -a "$LOG"
    return 0
}

say "=== pair run $STAMP: two single-task SmolVLA models, $STEPS steps each ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

# Preflight the raw datasets before trimming, not after: trimming rewrites 76k
# frames, and camera keys and tensor shapes are the same either side of it. The
# whole point of the check is to happen before anything slow.
for ds in so101_v3 so101_color6; do
    say "preflight $ds"
    "$V/python" "$WIN/preflight_train.py" "$BASE/data/$ds" \
        --policy "$WIN/survey/models/smolvla_base" --rename "$RENAME" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "!!! preflight failed for $ds"; exit 3; }
done

for ds in so101_v3 so101_color6; do
    trim_if_needed "$ds" || { say "!!! trim failed for $ds"; exit 3; }
done

train_one so101_v3     smolvla_v3only_20k
a=$?
train_one so101_color6 smolvla_color6only_20k
b=$?

say "=== finished: v3only rc=$a  color6only rc=$b ==="
exit $(( a + b ))
