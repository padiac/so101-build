#!/usr/bin/env bash
# Diffusion policy on the 453-episode colour set. The control with no language.
#
# Why this run exists. Every colour result so far is confounded: SmolVLA cannot
# reliably reach and close on this task, so "it did not take the named cube" has
# two readings -- it ignored the word, or it aimed badly. The two cannot be
# separated while grasping itself is unreliable.
#
# Diffusion has no language input at all. On so101_v3 it is the only policy whose
# plans come close to the demonstrations (roughness 0.14 against 0.11, where
# SmolVLA at its best is 0.49). So:
#
#   it grasps a cube reliably   -> these recordings do contain a learnable grasp,
#                                  and what remains for colour is purely language
#   it does not                 -> the grasp itself is the problem and colour is
#                                  not worth another run until that is fixed
#
# Either way the answer costs one night and no new recordings.
#
# The instruction is ignored by construction, so which colour was asked for is
# irrelevant here; success is "a cube leaves the bowl".
#
# Settings follow the v3 diffusion run so the two are comparable: 240x320 with a
# 95% random crop (the defaults feed the recorded 640x480 through a resnet18,
# twice over for n_obs_steps=2 and once per camera, with no augmentation), batch
# 8, 100k steps to match act_v3rand_100k and diffusion_v3_100k. Measured rate on
# this card: 7.8 step/s, about 3.6 hours.
#
#   bash tools/train_color6_453_diffusion.sh
#   START_AT=02:00 STEPS=50000 bash tools/train_color6_453_diffusion.sh
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
DS=so101_color6_453_trim
STEPS=${STEPS:-100000}
BATCH=${BATCH:-8}
SAVE_EVERY=${SAVE_EVERY:-5000}
MAX_TRIES=${MAX_TRIES:-3}
FREE_MB=${FREE_MB:-4000}
START_AT=${START_AT:-}
RESIZE='[240,320]'

LOG=$BASE/logs/diffusion_color6_453_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== diffusion on $DS: $STEPS steps, batch $BATCH ==="
[ -d "$BASE/data/$DS" ] || { say "!!! $BASE/data/$DS missing"; exit 2; }
"$V/python" -c "import diffusers" 2>/dev/null || { say "!!! diffusers not installed"; exit 2; }

if [ -n "$START_AT" ]; then
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

out=$BASE/outputs/diffusion_${DS}_$(date +%Y%m%d_%H%M)
mlog=$BASE/logs/diffusion_${DS}_$(date +%Y%m%d_%H%M).log
rc=1
for (( try = 1; try <= MAX_TRIES; try++ )); do
    prev=$(last_step "$out" 2>/dev/null || echo 0)
    if [ ! -d "$out/checkpoints/last" ]; then
        say "=== attempt $try ($STEPS steps, batch $BATCH) ==="
        "$V/lerobot-train" \
            --policy.type=diffusion \
            --dataset.repo_id="local/$DS" \
            --dataset.root="$BASE/data/$DS" \
            --policy.device=cuda --policy.push_to_hub=false \
            --policy.resize_shape="$RESIZE" --policy.crop_ratio=0.95 \
            --output_dir="$out" --job_name="diffusion_color6_453" \
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

if [ "$rc" -ne 0 ]; then
    say "!!! FAILED"
    exit 1
fi

name=diffusion_color6_453_$(( STEPS / 1000 ))k
say "publishing -> policies/$name"
rm -rf "${WIN:?}/policies/$name"
cp -r "$out/checkpoints/last/pretrained_model" "$WIN/policies/$name"
say "done: $(grep -aoE 'step:[0-9]+K [^|]*loss:[0-9.]+' "$mlog" | tail -1)"
