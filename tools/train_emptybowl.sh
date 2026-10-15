#!/usr/bin/env bash
# Diffusion policy on so101_emptybowl: take all six cubes out of the bowl.
#
# Why this task. The colour task is stuck for a reason that is now measured and
# is not about training: the colour word carries no information in those
# recordings, and language-conditioned policies score 0-10% on colour selection
# on this robot in the published benchmark. What the 09-27 control did show is
# that the grasp itself is learnable, and that the one real execution problem
# left is reaching into clutter without knocking the neighbours.
#
# This task removes the unanswerable half and trains the hard half six times per
# episode. One fixed instruction, no selection, success is simply an empty bowl.
#
# Two things are cleaned before training:
#
#   * episode 11 is 409 frames against a median of 1434, a run that was stopped
#     a quarter of the way through. Six cubes cannot be removed in 13 seconds,
#     so it is a demonstration of giving up and it is dropped.
#   * the idle head at the start of each episode IS trimmed. On 2026-10-14 the
#     untrimmed diffusion model sat completely still on the robot. From the
#     resting start pose it planned 1.7 units of travel and moved in 8% of
#     samples; the trimmed model planned 54.4 and moved in 100%
#     (tools/policy/does_it_start.py). The cause is not L1 averaging, which is
#     what I had assumed limited this to ACT: the idle head's label is "keep
#     waiting", the arm at rest sees exactly those frames, and waiting does not
#     change what it sees -- a loop any policy conditioned on the current frame
#     falls into, whether it averages or samples. TRIM=0 skips it.
#
# Diffusion rather than SmolVLA: on the colour recordings its plans came out at
# roughness 0.122 against the demonstrations' 0.106, where SmolVLA's best was
# 0.786, and it has no language input to waste on a task with one instruction.
#
#   bash tools/train_emptybowl.sh
#   STEPS=50000 bash tools/train_emptybowl.sh
#   REPREP=1 bash tools/train_emptybowl.sh      # rebuild the prepared copy
#   RESUME_OUT=<outputs dir> bash tools/train_emptybowl.sh   # continue a dead run
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
SRC=so101_emptybowl
STEPS=${STEPS:-100000}
BATCH=${BATCH:-8}
SAVE_EVERY=${SAVE_EVERY:-5000}
MAX_TRIES=${MAX_TRIES:-3}
FREE_MB=${FREE_MB:-4000}
START_AT=${START_AT:-}
TRIM=${TRIM:-1}
EXCLUDE="[11]"
# The recording grows between runs, so the expected count is read from the
# Windows copy rather than hard-coded; the check still catches a bad copy.
RESIZE='[240,320]'

LOG=$BASE/logs/emptybowl_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait BEFORE preparing, not after. Preparation trims every video on the CPU
# for one to two hours, and when it ran first it held two cores busy all
# evening while the person was still using the machine. The episode count is
# read after the wait too, so episodes recorded in the meantime are included
# instead of tripping the count check.
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

EXPECT_RAW=$("$V/python" -c "import json;print(json.load(open('$WIN/datasets/$SRC/meta/info.json'))['total_episodes'])")
EXPECT_KEPT=$(( EXPECT_RAW - 1 ))
PREP=${SRC}_trim
[ "$TRIM" = "1" ] || PREP=${SRC}_notrim
say "=== emptybowl: $EXPECT_RAW recorded, $EXPECT_KEPT kept, TRIM=$TRIM, $STEPS steps ==="

count_episodes() {
    "$V/python" - "$1" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "meta" / "info.json"
print(json.loads(p.read_text())["total_episodes"] if p.exists() else -1)
PY
}

if [ -d "$BASE/data/$PREP" ] && [ "$(count_episodes "$BASE/data/$PREP")" = "$EXPECT_KEPT" ] \
   && [ "${REPREP:-0}" != "1" ]; then
    say "=== data already prepared, reusing it (REPREP=1 to rebuild) ==="
else
    say "=== preparing data (CPU only) ==="
    rm -rf "$BASE/data/${SRC}_raw" "$BASE/data/${SRC}_kept" "$BASE/data/$PREP"
    cp -r "$WIN/datasets/$SRC" "$BASE/data/${SRC}_raw"
    rm -rf "$BASE/data/${SRC}_raw/clips"
    n=$(count_episodes "$BASE/data/${SRC}_raw")
    say "copied: $n episodes"
    [ "$n" = "$EXPECT_RAW" ] || { say "!!! expected $EXPECT_RAW episodes, got $n"; exit 3; }

    say "dropping the episode that stops a quarter of the way through: $EXCLUDE"
    "$V/lerobot-edit-dataset" \
        --repo_id "local/${SRC}_raw" --root "$BASE/data/${SRC}_raw" \
        --new_repo_id "local/${SRC}_kept" --new_root "$BASE/data/${SRC}_kept" \
        --operation.type delete_episodes \
        --operation.episode_indices "$EXCLUDE" 2>&1 | tail -5 | tee -a "$LOG"
    n=$(count_episodes "$BASE/data/${SRC}_kept")
    [ "$n" = "$EXPECT_KEPT" ] || { say "!!! expected $EXPECT_KEPT after the drop, got $n"; exit 3; }
    say "after drop: $n episodes"
    rm -rf "$BASE/data/${SRC}_raw"

    if [ "$TRIM" = "1" ]; then
        say "trimming the still head off each episode"
        cp -f "$WIN/trim_dataset.py" "$BASE/trim_dataset.py"
        "$V/python" "$BASE/trim_dataset.py" --src "$BASE/data/${SRC}_kept" \
            --dst "$BASE/data/$PREP" --repo-id "local/$PREP" --force 2>&1 | tail -3 | tee -a "$LOG"
        [ -d "$BASE/data/$PREP" ] || { say "!!! trim failed"; exit 3; }
        rm -rf "$BASE/data/${SRC}_kept"
    else
        say "not trimming (TRIM=0)"
        mv "$BASE/data/${SRC}_kept" "$BASE/data/$PREP"
    fi
    say "prepared: $(count_episodes "$BASE/data/$PREP") episodes in $PREP"
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

# The episode count goes in the name so a bigger recording never overwrites the
# model trained on the smaller one -- they are the comparison.
name=diffusion_emptybowl${EXPECT_KEPT}_$(( STEPS / 1000 ))k
[ "$TRIM" = "1" ] && name=${name}_trim
say "publishing -> policies/$name"
rm -rf "${WIN:?}/policies/$name"
cp -r "$out/checkpoints/last/pretrained_model" "$WIN/policies/$name"
say "done: $(grep -aoE 'step:[0-9]+K [^|]*loss:[0-9.]+' "$mlog" | tail -1)"
