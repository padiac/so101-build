#!/usr/bin/env bash
# The 453-episode rerun of the colour / pooled pair, with a validation split.
#
# 2026-09-12 extended so101_color6 from 304 to 453 episodes (about 90 per colour).
# Same two models as train_color6_304.sh:
#
#   A  so101_color6_453   the real task, five colour instructions
#   B  so101_c6pool_453   the same clips behind one instruction, colour ignored
#
# What is different from the 304 run, and why:
#
# 1. Four episodes are dropped: 324, 354, 393, 402. Each is 25 s with no arm or
#    gripper motion at all -- pure "sit still" demonstrations, the exact failure
#    trim_dataset.py exists to prevent. Found with an audit of the new batch on
#    2026-09-13. The drop happens on the WSL copy only; datasets/so101_color6 on
#    the Windows side is left untouched.
#
# 2. 40000 steps with a 10% validation split. The mix1 100k run (2026-09-06, 280
#    episodes) had its lowest eval loss at step 4000 (0.181) and rose steadily
#    after -- 0.219 at 20k, 0.269 at 40k, 0.385 at 100k. More data should move
#    that minimum later, so this run measures it instead of guessing: eval every
#    2000 steps, a checkpoint every 2000 steps, and the best one published too.
#
# 3. The LR schedule is stretched to the full run. smolvla's cosine decay is
#    30000 steps and lerobot only auto-scales it when the run is SHORTER, so a
#    40000-step run would otherwise sit at the floor LR for its last 10000 steps.
#
# The two held-out sets are identical, which is not obvious. lerobot holds out the
# last 10% of episodes PER TASK, grouping by meta/episodes "tasks". pool_tasks.py
# rewrites tasks.parquet and the task_index column but not meta/episodes, so the
# pooled copy still groups by colour for the split, while the text the model is
# fed comes from task_index -> tasks.parquet and is the single pooled instruction.
# Without that, the pooled model would hold out the last 45 episodes overall --
# 423-452 are all blue -- and train on far less blue than the colour model.
# Check the "Train/eval split" line of both runs: same counts, 5 tasks each.
#
#   bash tools/train_color6_453.sh            # usually via a detached snapshot
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
V=$BASE/.venv/bin
SRC=so101_color6
TAG=453
STEPS=${STEPS:-40000}
WHICH=${WHICH:-both}
BATCH=${BATCH:-8}
MAX_TRIES=${MAX_TRIES:-4}
FREE_MB=${FREE_MB:-3000}
EXCLUDE="[324, 354, 393, 402]"
EXPECT_EPISODES=449
EVAL_SPLIT=0.1
EVAL_EVERY=2000
POOL_TASK="Pick a cube out of the bowl and put it on the table."

RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

LOG=$BASE/logs/color6_${TAG}_$(date +%Y%m%d_%H%M).log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

count_episodes() { "$V/python" -c "import json,sys; print(json.load(open(sys.argv[1]+'/meta/info.json'))['total_episodes'])" "$1"; }

if [ -d "$BASE/data/${SRC}_${TAG}_trim" ] && [ -d "$BASE/data/so101_c6pool_${TAG}_trim" ] && [ "${REPREP:-0}" != "1" ]; then
    say "=== data already prepared, reusing it (REPREP=1 to rebuild) ==="
else
    say "=== preparing data (CPU only) ==="
    cp -f "$WIN/preflight_train.py" "$BASE/preflight_train.py" 2>/dev/null || true

    say "copying $SRC from the Windows side"
    rm -rf "$BASE/data/${SRC}_${TAG}_raw" "$BASE/data/${SRC}_${TAG}"
    cp -r "$WIN/datasets/$SRC" "$BASE/data/${SRC}_${TAG}_raw"
    rm -rf "$BASE/data/${SRC}_${TAG}_raw/clips"
    n=$(count_episodes "$BASE/data/${SRC}_${TAG}_raw")
    say "copied: $n episodes"
    [ "$n" = "453" ] || { say "!!! expected 453 episodes in the source, got $n"; exit 3; }

    say "dropping the no-motion episodes $EXCLUDE"
    "$V/lerobot-edit-dataset" \
        --repo_id "local/${SRC}_${TAG}_raw" --root "$BASE/data/${SRC}_${TAG}_raw" \
        --new_repo_id "local/${SRC}_${TAG}" --new_root "$BASE/data/${SRC}_${TAG}" \
        --operation.type delete_episodes \
        --operation.episode_indices "$EXCLUDE" 2>&1 | tail -5 | tee -a "$LOG"
    n=$(count_episodes "$BASE/data/${SRC}_${TAG}")
    [ "$n" = "$EXPECT_EPISODES" ] || { say "!!! expected $EXPECT_EPISODES after the drop, got $n"; exit 3; }
    say "after drop: $n episodes"
    rm -rf "$BASE/data/${SRC}_${TAG}_raw"

    say "preflight"
    "$V/python" "$WIN/preflight_train.py" "$BASE/data/${SRC}_${TAG}" \
        --policy "$WIN/survey/models/smolvla_base" --rename "$RENAME" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "!!! preflight failed"; exit 3; }

    say "trimming the still head off each episode"
    cp -f "$WIN/trim_dataset.py" "$BASE/trim_dataset.py"
    "$V/python" "$BASE/trim_dataset.py" --src "$BASE/data/${SRC}_${TAG}" \
        --dst "$BASE/data/${SRC}_${TAG}_trim" --repo-id "local/${SRC}_${TAG}_trim" --force 2>&1 | tail -3 | tee -a "$LOG"
    [ -d "$BASE/data/${SRC}_${TAG}_trim" ] || { say "!!! trim failed"; exit 3; }

    say "building the colour-agnostic copy"
    cp -f "$WIN/tools/dataset/pool_tasks.py" "$BASE/pool_tasks.py"
    "$V/python" "$BASE/pool_tasks.py" --src "$BASE/data/${SRC}_${TAG}_trim" \
        --dst "$BASE/data/so101_c6pool_${TAG}_trim" --task "$POOL_TASK" --force \
        2>&1 | tail -4 | tee -a "$LOG"
    [ -d "$BASE/data/so101_c6pool_${TAG}_trim" ] || { say "!!! pooling failed"; exit 3; }
    say "data ready"
fi

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
    local ds=$1 out_name=$2 out try prev now rc mlog best
    out=$BASE/outputs/smolvla_${ds}_$(date +%Y%m%d_%H%M)
    mlog=$BASE/logs/smolvla_${ds}_$(date +%Y%m%d_%H%M).log
    for (( try = 1; try <= MAX_TRIES; try++ )); do
        prev=$(last_step "$out" 2>/dev/null || echo 0)
        if [ ! -d "$out/checkpoints/last" ]; then
            say "=== train $ds  attempt $try  ($STEPS steps, batch $BATCH, eval_split $EVAL_SPLIT) ==="
            "$V/lerobot-train" \
                --policy.path=lerobot/smolvla_base \
                --dataset.repo_id="local/${ds}_trim" \
                --dataset.root="$BASE/data/${ds}_trim" \
                --rename_map="$RENAME" \
                --policy.device=cuda --policy.push_to_hub=false \
                --policy.scheduler_decay_steps="$STEPS" \
                --output_dir="$out" --job_name="smolvla_${ds}" \
                --steps="$STEPS" --save_freq="$EVAL_EVERY" --log_freq=200 \
                --dataset.eval_split="$EVAL_SPLIT" --eval_steps="$EVAL_EVERY" --max_eval_samples=512 \
                --batch_size="$BATCH" --num_workers=4 --prefetch_factor=2 \
                --wandb.enable=false 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
        else
            say "=== resume $ds  attempt $try  (from $(last_step "$out")) ==="
            "$V/lerobot-train" --resume=true \
                --config_path="$out/checkpoints/last/pretrained_model" 2>&1 | tee -a "$mlog" "$LOG" > /dev/null
        fi
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] && break
        now=$(last_step "$out" 2>/dev/null || echo 0)
        say "$ds exited $rc at step $now (was $prev)"
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
    say "$ds split: $(grep -aoE 'Train/eval split: [^(]*\([^)]*\)' "$mlog" | tail -1)"

    say "publishing last -> policies/$out_name"
    rm -rf "${WIN:?}/policies/$out_name"
    cp -r "$out/checkpoints/last/pretrained_model" "$WIN/policies/$out_name"

    # The step with the lowest eval loss that also has a checkpoint.
    best=$(grep -aoE 'step [0-9]+: eval_loss=[0-9.]+' "$mlog" \
        | sed -E 's/step ([0-9]+): eval_loss=/\1 /' | sort -k2 -g \
        | while read -r s l; do
              d=$(printf '%06d' "$s")
              [ -d "$out/checkpoints/$d" ] && { echo "$s $l"; break; }
          done)
    if [ -n "$best" ]; then
        read -r bs bl <<< "$best"
        bk=$(( bs / 1000 ))k
        say "best eval loss $bl at step $bs -> policies/${out_name}_best${bk}"
        rm -rf "${WIN:?}/policies/${out_name}_best"*
        cp -r "$out/checkpoints/$(printf '%06d' "$bs")/pretrained_model" "$WIN/policies/${out_name}_best${bk}"
    fi
    grep -aoE 'step [0-9]+: eval_loss=[0-9.]+' "$mlog" > "$out/eval_curve.txt"
    say "$ds done: $(grep -aoE 'step:[0-9]+K [^|]*loss:[0-9.]+' "$mlog" | tail -1)"
    return 0
}

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
