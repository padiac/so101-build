#!/usr/bin/env bash
# Merge -> copy -> trim -> finetune SmolVLA, as one detached job.
#
# Why merge: capability accumulates in the DATA, not in the weights. Finetuning
# on task B alone forgets task A, so every new task is trained on the union of
# everything recorded so far. Retraining cost is flat (steps are steps), so the
# only growing cost is disk.
#
# The two sets pair unusually well: so101_v3 puts the yellow cube INTO the bowl,
# so101_color6 takes a named cube OUT of it. The word "yellow" appears in both
# with opposite directions, so the policy cannot succeed by spotting the colour
# word alone -- it has to read the whole instruction.
#
# The delay is a plain sleep rather than a scheduler because the job must survive
# the shell that armed it.
set -uo pipefail

DELAY=${DELAY:-0}
WIN=/mnt/e/Repo/so101-build
BASE=/home/padiac/lerobot-train
DS=${DS:-so101_mix1}                 # merged dataset name
PARTS=${PARTS:-"so101_v3 so101_color6"}

if [ "$DELAY" -gt 0 ]; then
    echo "=== armed $(date); starting in $((DELAY/60)) min ==="
    sleep "$DELAY"
fi

# Preflight before anything expensive. On 2026-09-05 the merge and trim ran for
# ninety minutes and training then died in its first second because smolvla_base
# names its cameras camera1/2/3 while these datasets name them top/wrist. That is
# a ten-second check, so it happens first now.
RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
echo "=== preflight  $(date) ==="
for p in $PARTS; do
    "$BASE/.venv/bin/python" "$WIN/preflight_train.py" "$WIN/datasets/$p"         --policy "$WIN/survey/models/smolvla_base" --rename "$RENAME" || {
            echo "PREFLIGHT FAILED on $p -- refusing to spend the hour"; exit 3; }
done

echo "=== copy source datasets into ext4  $(date) ==="
roots=""
ids=""
for p in $PARTS; do
    rm -rf "$BASE/data/$p"
    cp -r "$WIN/datasets/$p" "$BASE/data/"
    rm -rf "$BASE/data/$p/clips"     # review clips are not training data
    roots="$roots'$BASE/data/$p',"
    ids="$ids'local/$p',"
done

echo "=== merge into $DS  $(date) ==="
rm -rf "$BASE/data/$DS"
"$BASE/.venv/bin/lerobot-edit-dataset" \
  --new_repo_id "local/$DS" --new_root "$BASE/data/$DS" \
  --operation.type merge \
  --operation.repo_ids "[${ids%,}]" \
  --operation.roots "[${roots%,}]" || exit 1

echo "=== trim the still period at the head  $(date) ==="
cp "$WIN/trim_dataset.py" "$BASE/trim_dataset.py"
"$BASE/.venv/bin/python" "$BASE/trim_dataset.py" \
  --src "$BASE/data/$DS" --dst "$BASE/data/${DS}_trim" --force || exit 1

echo "=== finetune SmolVLA from lerobot/smolvla_base  $(date) ==="
STAMP=$(date +%Y%m%d_%H%M)
OUT=$BASE/outputs/smolvla_${DS}_$STAMP
LOG=$BASE/logs/smolvla_${DS}_$STAMP.log
mkdir -p "$BASE/logs"
ln -sfn "$LOG" "$BASE/logs/current.log"

# batch 16 measured at 6.4 GB of the 3080's 10.2 GB, ~10 samples/s -- the largest
# batch with real headroom (32 peaked at 9.3 GB, too close after this machine's
# out-of-memory history).
set +e
"$BASE/.venv/bin/lerobot-train" \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/${DS}_trim \
  --dataset.root=$BASE/data/${DS}_trim \
  --rename_map="$RENAME" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=smolvla_${DS}_$STAMP \
  --steps=${STEPS:-20000} \
  --save_freq=2000 \
  --log_freq=200 \
  --batch_size=${BATCH:-16} \
  --num_workers=4 --prefetch_factor=2 \
  --wandb.enable=false 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "=== exit $rc  $(date) ===" | tee -a "$LOG"
exit "$rc"
