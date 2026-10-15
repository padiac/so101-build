#!/usr/bin/env bash
# Can this card train SmolVLA with the vision-language model unfrozen?
#
# Every run so far used the recipe's default, train_expert_only=true: the 450M
# backbone is frozen and only the ~100M action expert is updated. Unfreezing
# means optimizer state, gradients and activations for the whole model, which on
# a 10 GB card is the question, not a detail.
#
# Twenty steps is enough to see peak memory or an out-of-memory.
#
#   bash tools/unfreeze_probe.sh            # batch 1 2 4
#   SIZES="1 2" bash tools/unfreeze_probe.sh
set -uo pipefail
BASE=/home/padiac/lerobot-train
SIZES=${SIZES:-"1 2 4"}
STEPS=${STEPS:-20}
DS=${DS:-so101_v3_trim}
cd "$BASE" || exit 1

RENAME='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

printf '%8s  %10s  %10s  %s\n' batch step/s mem_gb result
for b in $SIZES; do
    out=$BASE/outputs/unfreeze_${b}_$(date +%H%M%S)
    log=$BASE/logs/unfreeze_${b}.log
    "$BASE/.venv/bin/lerobot-train" \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id="local/$DS" --dataset.root="$BASE/data/$DS" \
        --rename_map="$RENAME" \
        --policy.device=cuda --policy.push_to_hub=false \
        --policy.train_expert_only=false \
        --steps="$STEPS" --save_freq=1000000 --log_freq=10 \
        --batch_size="$b" --num_workers=4 \
        --output_dir="$out" > "$log" 2>&1
    rc=$?
    line=$(grep -aoE 'smp/s:[0-9]+ mem_gb:[0-9.]+' "$log" | tail -1)
    smp=$(echo "$line" | grep -oE 'smp/s:[0-9]+' | cut -d: -f2)
    mem=$(echo "$line" | grep -oE 'mem_gb:[0-9.]+' | cut -d: -f2)
    if [ "$rc" -eq 0 ] && [ -n "$smp" ]; then
        printf '%8s  %10s  %10s  ok\n' "$b" "$(awk -v s="$smp" -v b="$b" 'BEGIN{printf "%.2f", s/b}')" "$mem"
    elif grep -aq "OutOfMemoryError\|out of memory" "$log"; then
        printf '%8s  %10s  %10s  out of memory\n' "$b" - -
    else
        printf '%8s  %10s  %10s  failed rc=%s (%s)\n' "$b" - - "$rc" "$log"
    fi
    rm -rf "$out"
done
