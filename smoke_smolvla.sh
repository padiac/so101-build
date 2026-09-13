#!/usr/bin/env bash
# Does SmolVLA fit on the 10 GB RTX 3080? Twenty steps is enough to see peak
# memory, and it decides whether stage-2 multi-task work needs Colab at all.
set -uo pipefail
B=/home/padiac/lerobot-train
BATCH=${BATCH:-2}
STAMP=$(date +%H%M%S)
LOG=$B/logs/smolvla_probe_${BATCH}_$STAMP.log
echo "batch=$BATCH  log=$LOG"
"$B/.venv/bin/lerobot-train" \
  --dataset.repo_id=local/so101_v3_trim \
  --dataset.root=$B/data/so101_v3_trim \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=$B/outputs/probe_smolvla_${BATCH}_$STAMP \
  --job_name=probe_smolvla \
  --steps=20 --save_freq=1000000 --log_freq=5 \
  --batch_size=$BATCH --num_workers=2 --prefetch_factor=2 \
  --wandb.enable=false > "$LOG" 2>&1 &
TRAIN=$!
PEAK=0
while kill -0 $TRAIN 2>/dev/null; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$U" -gt "$PEAK" ] && PEAK=$U
    sleep 2
done
wait $TRAIN; RC=$?
echo "exit $RC   peak GPU $PEAK MiB / 10240"
grep -E "num_learnable_params|num_total_params|Error|error|out of memory|Traceback" "$LOG" | head -6
tr '\r' '\n' < "$LOG" | grep -E "^Training:" | tail -1
