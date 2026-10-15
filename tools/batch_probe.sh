#!/usr/bin/env bash
# What batch size does SmolVLA actually fit on this card, and how fast?
#
# Every SmolVLA here was trained at batch 8, not by choice: on 2026-09-08 batch
# 16 died three times in the first forward pass. The paper's recipe is 64. Small
# batches mean noisy gradients, which is one of the few untested explanations for
# why these models plan jagged trajectories.
#
# lerobot 0.6.1 has no gradient accumulation, so the only levers are the batch
# itself and mixed precision. Sixty steps each is enough to see peak memory and a
# settled rate; the throwaway outputs go to /tmp of the WSL instance.
#
#   bash tools/batch_probe.sh                 # 8 16 24 32, with AMP
#   AMP=false SIZES="8 16" bash tools/batch_probe.sh
set -uo pipefail
BASE=/home/padiac/lerobot-train
SIZES=${SIZES:-"8 16 24 32"}
AMP=${AMP:-true}
STEPS=${STEPS:-60}
DS=${DS:-so101_v3_trim}
cd "$BASE" || exit 1

printf '%8s  %10s  %10s  %s\n' batch step/s mem_gb result
for b in $SIZES; do
    out=$BASE/outputs/batchprobe_${b}_$(date +%H%M%S)
    log=$BASE/logs/batchprobe_${b}.log
    "$BASE/.venv/bin/lerobot-train" \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id="local/$DS" --dataset.root="$BASE/data/$DS" \
        --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
        --policy.device=cuda --policy.push_to_hub=false --policy.use_amp="$AMP" \
        --steps="$STEPS" --save_freq=1000000 --log_freq=20 \
        --batch_size="$b" --num_workers=4 \
        --output_dir="$out" > "$log" 2>&1
    rc=$?
    # The last logged line carries both the rate and the peak memory lerobot saw.
    line=$(grep -aoE 'smp/s:[0-9]+ mem_gb:[0-9.]+' "$log" | tail -1)
    smp=$(echo "$line" | grep -oE 'smp/s:[0-9]+' | cut -d: -f2)
    mem=$(echo "$line" | grep -oE 'mem_gb:[0-9.]+' | cut -d: -f2)
    if [ "$rc" -eq 0 ] && [ -n "$smp" ]; then
        printf '%8s  %10s  %10s  ok\n' "$b" "$(awk -v s="$smp" -v b="$b" 'BEGIN{printf "%.2f", s/b}')" "$mem"
    elif grep -aq "OutOfMemoryError\|out of memory" "$log"; then
        printf '%8s  %10s  %10s  out of memory\n' "$b" - -
    else
        printf '%8s  %10s  %10s  failed rc=%s (see %s)\n' "$b" - - "$rc" "$log"
    fi
    rm -rf "$out"
done
