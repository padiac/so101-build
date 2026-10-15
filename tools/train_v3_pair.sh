#!/usr/bin/env bash
# SmolVLA at double the steps, and a diffusion policy, both on so101_v3.
#
# On the robot ACT beats SmolVLA on this task even though the data is identical:
# act_v3rand_100k finishes cleanly, smolvla_v3only_20k finishes but trembles and
# keeps stirring afterwards. Two things could explain it, and this run separates
# them.
#
#   A  smolvla_v3only_40k   the same recipe as the 20k model, twice the steps.
#                           If the gap is that SmolVLA was simply trained less
#                           than ACT's 100k, this closes it.
#   B  diffusion_v3_100k    a different policy class at ACT's own step count.
#                           If a second non-VLA model also beats SmolVLA here,
#                           the gap is about the policy class, not the budget.
#
# ACT already exists at 100k on this data, so the three are directly comparable.
#
# Why these numbers:
#
# * 40000 steps for SmolVLA because 20000 is what the current model had, and the
#   colour runs showed within-plan roughness still falling between 20k and 40k
#   before flattening. Twice is the cheapest step that could show the effect.
# * 100000 for diffusion to match act_v3rand_100k exactly.
# * No eval split, unlike the 453 colour runs. smolvla_v3only_20k and
#   act_v3rand_100k were both trained on all 130 episodes, and holding 10% out
#   now would mean the comparison changed two things at once.
# * Diffusion resizes to 240x320 with a 95% random crop. Its defaults feed the
#   recorded 640x480 straight into a resnet18, twice over for n_obs_steps=2 and
#   once per camera, with no augmentation.
#
# Measured before scheduling, not estimated: diffusion runs at 7.8 step/s and
# 5.45 GB (tools/diffusion_probe.sh, 200 steps, 2026-09-23), so 100k is about
# 3.6 h. SmolVLA ran at 4.4 step/s in the earlier pair run, so 40k is about 2.5 h.
# Together about six hours.
#
#   bash tools/train_v3_pair.sh                  # now
#   START_AT=02:30 bash tools/train_v3_pair.sh   # at the next 02:30
#   WHICH=diffusion bash tools/train_v3_pair.sh  # one arm only
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
DS=so101_v3
SMOL_STEPS=${SMOL_STEPS:-40000}
DIFF_STEPS=${DIFF_STEPS:-100000}
WHICH=${WHICH:-both}
BATCH=${BATCH:-8}
SAVE_EVERY=${SAVE_EVERY:-5000}
MAX_TRIES=${MAX_TRIES:-4}
FREE_MB=${FREE_MB:-3000}
START_AT=${START_AT:-}
RESIZE='[240,320]'

RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

LOG=$BASE/logs/v3pair_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== v3 pair: smolvla ${SMOL_STEPS}, diffusion ${DIFF_STEPS}, batch ${BATCH} ==="

[ -d "$BASE/data/${DS}_trim" ] || { say "!!! $BASE/data/${DS}_trim is missing"; exit 2; }
"$V/python" -c "import diffusers" 2>/dev/null || { say "!!! diffusers not installed"; exit 2; }
say "data and diffusers present"

if [ -n "$START_AT" ]; then
    # The NEXT occurrence of that clock time, but never a wait longer than
    # MAX_WAIT_H. Rolling unconditionally to tomorrow once put a run 22 hours
    # out; refusing to roll at all once started a 00:30 job at 21:32 the evening
    # before. The cap separates the two cases without anyone having to say which
    # day was meant.
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

# A deployment session on the Windows side holds about 3 GB of the 10, and so
# does a video playing in a browser -- that is what made the first diffusion
# probe die with 3.37 GB free.
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

train_one() {          # $1 = arm name (smolvla|diffusion), $2 = steps, $3 = publish name
    local arm=$1 steps=$2 out_name=$3 out try prev now rc mlog stamp
    stamp=$(date +%Y%m%d_%H%M)
    out=$BASE/outputs/${arm}_${DS}_${stamp}
    mlog=$BASE/logs/${arm}_${DS}_${stamp}.log
    for (( try = 1; try <= MAX_TRIES; try++ )); do
        prev=$(last_step "$out" 2>/dev/null || echo 0)
        if [ ! -d "$out/checkpoints/last" ]; then
            say "=== train $arm  attempt $try  ($steps steps, batch $BATCH) ==="
            if [ "$arm" = "smolvla" ]; then
                "$V/lerobot-train" \
                    --policy.path=lerobot/smolvla_base \
                    --dataset.repo_id="local/${DS}_trim" \
                    --dataset.root="$BASE/data/${DS}_trim" \
                    --rename_map="$RENAME" \
                    --policy.device=cuda --policy.push_to_hub=false \
                    --policy.scheduler_decay_steps="$steps" \
                    --output_dir="$out" --job_name="${arm}_${DS}" \
                    --steps="$steps" --save_freq="$SAVE_EVERY" --log_freq=200 \
                    --batch_size="$BATCH" --num_workers=4 --prefetch_factor=2 \
                    --wandb.enable=false 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
            else
                "$V/lerobot-train" \
                    --policy.type=diffusion \
                    --dataset.repo_id="local/${DS}_trim" \
                    --dataset.root="$BASE/data/${DS}_trim" \
                    --policy.device=cuda --policy.push_to_hub=false \
                    --policy.resize_shape="$RESIZE" --policy.crop_ratio=0.95 \
                    --output_dir="$out" --job_name="${arm}_${DS}" \
                    --steps="$steps" --save_freq="$SAVE_EVERY" --log_freq=200 \
                    --batch_size="$BATCH" --num_workers=4 --prefetch_factor=2 \
                    --wandb.enable=false 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
            fi
        else
            say "=== resume $arm  attempt $try  (from $(last_step "$out")) ==="
            "$V/lerobot-train" --resume=true \
                --config_path="$out/checkpoints/last/pretrained_model" 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
        fi
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] && break
        now=$(last_step "$out" 2>/dev/null || echo 0)
        say "$arm exited $rc at step $now (was $prev)"
        if [ "$try" -gt 1 ] && [ "$now" != "0" ] && [ "$now" = "$prev" ]; then
            say "no progress since the last attempt -- a real fault"
            break
        fi
        say "retrying in 120s"
        sleep 120
    done
    if [ "$rc" -ne 0 ]; then
        say "!!! $arm FAILED"
        return 1
    fi
    say "publishing last -> policies/$out_name"
    rm -rf "${WIN:?}/policies/$out_name"
    cp -r "$out/checkpoints/last/pretrained_model" "$WIN/policies/$out_name"
    say "$arm done: $(grep -aoE 'step:[0-9]+K [^|]*loss:[0-9.]+' "$mlog" | tail -1)"
    return 0
}

a=0; b=0
case "$WHICH" in
    both|smolvla)   train_one smolvla   "$SMOL_STEPS" "smolvla_v3only_$(( SMOL_STEPS / 1000 ))k"; a=$? ;;
esac
case "$WHICH" in
    both|diffusion) train_one diffusion "$DIFF_STEPS" "diffusion_v3_$(( DIFF_STEPS / 1000 ))k"; b=$? ;;
esac
say "=== finished ($WHICH): smolvla rc=$a  diffusion rc=$b ==="
exit $(( a + b ))
