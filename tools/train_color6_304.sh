#!/usr/bin/env bash
# The 304-episode rerun: same two questions, twice the demonstrations.
#
# so101_color6 was 150 episodes, 30 per colour, and the model trained on it did
# not use the colour word at all -- changing the word moved its plan by less than
# the noise of asking twice with the same word (ratio 0.80 from
# tools/policy/reads_the_word.py). 2026-09-10 doubled it to 304, about 60 per
# colour.
#
# Two models, because two different things could have been the bottleneck:
#
#   A  so101_color6_304   the real task, five colour instructions
#   B  so101_c6pool_304   the same clips behind one instruction, colour ignored
#
# A answers "is 60 per colour enough to make the word matter". B answers "is the
# motion learnable at all", with the ambiguity of six cubes still in it.
#
# Everything but the data matches the earlier models: same base checkpoint, same
# rename map, batch 8, 20000 steps.
#
# Data preparation is CPU work and runs immediately. Training waits for the hour
# it was asked for, and for the GPU.
#
#   START_AT=02:30 bash tools/train_color6_304.sh
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
SRC=so101_color6                 # the Windows-side dataset, 304 episodes
TAG=304                          # kept in every name, so the 150-episode
                                 # datasets and models stay distinguishable
STEPS=${STEPS:-20000}
# Which of the two to train: both, colour, or pool. A 100000-step run is nearly
# seven hours, so it is usually one at a time.
WHICH=${WHICH:-both}
BATCH=${BATCH:-8}
MAX_TRIES=${MAX_TRIES:-4}
START_AT=${START_AT:-}
FREE_MB=${FREE_MB:-3000}
POOL_TASK="Pick a cube out of the bowl and put it on the table."

RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

LOG=$BASE/logs/color6_${TAG}_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if [ -d "$BASE/data/${SRC}_${TAG}_trim" ] &&    [ -d "$BASE/data/so101_c6pool_${TAG}_trim" ] && [ "${REPREP:-0}" != "1" ]; then
    say "=== data already prepared, reusing it (REPREP=1 to rebuild) ==="
    say "  ${SRC}_${TAG}_trim and so101_c6pool_${TAG}_trim"
    PREP=0
else
    PREP=1
fi

if [ "$PREP" = "1" ]; then
say "=== preparing data (CPU only, no GPU needed) ==="

# Preflight the raw dataset before the slow part, not after. Trimming rewrites
# 148870 frames of video; a camera-key mismatch found afterwards has already cost
# ninety minutes once.
cp -f "$WIN/preflight_train.py" "$BASE/preflight_train.py" 2>/dev/null || true
say "copying $SRC from the Windows side"
rm -rf "$BASE/data/${SRC}_${TAG}"
cp -r "$WIN/datasets/$SRC" "$BASE/data/${SRC}_${TAG}"
rm -rf "$BASE/data/${SRC}_${TAG}/clips"          # review clips are not training data

say "preflight"
"$V/python" "$WIN/preflight_train.py" "$BASE/data/${SRC}_${TAG}" \
    --policy "$WIN/survey/models/smolvla_base" --rename "$RENAME" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "!!! preflight failed"; exit 3; }

say "trimming the still head off each episode"
cp -f "$WIN/trim_dataset.py" "$BASE/trim_dataset.py"
"$V/python" "$BASE/trim_dataset.py" --src "$BASE/data/${SRC}_${TAG}" \
    --dst "$BASE/data/${SRC}_${TAG}_trim" --force 2>&1 | tail -3 | tee -a "$LOG"
[ -d "$BASE/data/${SRC}_${TAG}_trim" ] || { say "!!! trim failed"; exit 3; }

say "building the colour-agnostic copy"
cp -f "$WIN/tools/dataset/pool_tasks.py" "$BASE/pool_tasks.py"
"$V/python" "$BASE/pool_tasks.py" --src "$BASE/data/${SRC}_${TAG}_trim" \
    --dst "$BASE/data/so101_c6pool_${TAG}_trim" --task "$POOL_TASK" --force \
    2>&1 | tail -4 | tee -a "$LOG"
[ -d "$BASE/data/so101_c6pool_${TAG}_trim" ] || { say "!!! pooling failed"; exit 3; }
say "data ready"
fi

if [ -n "$START_AT" ]; then
    # The NEXT occurrence of that clock time, but never a wait longer than
    # MAX_WAIT_H. Both halves were learned the hard way. Rolling unconditionally
    # to tomorrow put a run 22 hours out because two hours of data preparation
    # ended just after the target. Refusing to roll at all then started a run
    # meant for 00:30 at 21:32 the evening before, because 00:30 had "already
    # passed" today. The cap distinguishes them without anyone having to think
    # about which day was meant.
    MAX_WAIT_H=${MAX_WAIT_H:-12}
    target=$(date -d "today $START_AT" +%s 2>/dev/null)
    now=$(date +%s)
    [ "$target" -le "$now" ] && target=$(date -d "tomorrow $START_AT" +%s)
    wait_s=$(( target - now ))
    if [ "$wait_s" -gt $(( MAX_WAIT_H * 3600 )) ]; then
        say "$START_AT is $(( wait_s / 3600 ))h away, past the ${MAX_WAIT_H}h cap"
        say "  (the work ran long and overshot it) -- starting now instead"
    else
        say "waiting for $START_AT on $(date -d "@$target" '+%m-%d') ""(in $(( wait_s / 60 )) min)"
        while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done
    fi
fi

# A deployment session on the Windows side holds about 3 GB of the 10, and this
# machine already refuses to train at batch 16 with room to spare.
waited=0
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt "$FREE_MB" ] && break
    [ "$waited" -eq 0 ] && say "GPU busy (${used} MiB); waiting"
    waited=$((waited + 1))
    [ "$waited" -gt 360 ] && { say "GPU still busy after 3h; starting anyway"; break; }
    sleep 30
done

last_step() {
    local d
    d=$(ls -d "$1"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1) || return 1
    [ -n "$d" ] && basename "$d" || echo 0
}

train_one() {          # $1 = dataset dir name (without _trim), $2 = publish name
    local ds=$1 out_name=$2 out try prev now rc
    out=$BASE/outputs/smolvla_${ds}_$(date +%Y%m%d_%H%M)
    for (( try = 1; try <= MAX_TRIES; try++ )); do
        prev=$(last_step "$out" 2>/dev/null || echo 0)
        if [ ! -d "$out/checkpoints/last" ]; then
            say "=== train $ds  attempt $try  ($STEPS steps, batch $BATCH) ==="
            "$V/lerobot-train" \
                --policy.path=lerobot/smolvla_base \
                --dataset.repo_id="local/${ds}_trim" \
                --dataset.root="$BASE/data/${ds}_trim" \
                --rename_map="$RENAME" \
                --policy.device=cuda --policy.push_to_hub=false \
                --output_dir="$out" --job_name="smolvla_${ds}" \
                --steps="$STEPS" --save_freq=2000 --log_freq=200 \
                --batch_size="$BATCH" --num_workers=4 --prefetch_factor=2 \
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
        # Only a repeat failure at the SAME checkpoint is a real fault; written as
        # a bare equality this also fired before any checkpoint existed and
        # abandoned a model over one transient CUDA error.
        if [ "$try" -gt 1 ] && [ "$now" != "0" ] && [ "$now" = "$prev" ]; then
            say "no progress since the last attempt -- a real fault"
            break
        fi
        say "retrying in 120s"
        sleep 120
    done
    if [ "$rc" -ne 0 ]; then
        say "!!! $ds FAILED"
        return 1
    fi
    say "publishing -> policies/$out_name"
    rm -rf "${WIN:?}/policies/$out_name"
    cp -r "$out/checkpoints/last/pretrained_model" "$WIN/policies/$out_name"
    say "$ds done: $(grep -aoE 'step:[0-9]+K [^|]*loss:[0-9.]+' "$LOG" | tail -1)"
    return 0
}

# The published name carries the step count, so a 100000-step model never lands
# on top of the 20000-step one it is meant to be compared against.
KSTEPS=$(( STEPS / 1000 ))k
a=0; b=0
case "$WHICH" in
    both|colour) train_one "${SRC}_${TAG}"       "smolvla_color6_${TAG}_${KSTEPS}"; a=$? ;;
esac
case "$WHICH" in
    both|pool)   train_one "so101_c6pool_${TAG}" "smolvla_c6pool_${TAG}_${KSTEPS}"; b=$? ;;
esac
say "=== finished ($WHICH, $STEPS steps): colour rc=$a  pooled rc=$b ==="
exit $(( a + b ))
