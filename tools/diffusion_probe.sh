#!/usr/bin/env bash
# Two hundred steps of diffusion policy on so101_v3_trim, to measure what a full
# run would cost before one is scheduled overnight.
#
# The defaults feed the cameras at their recorded 640x480 through a resnet18,
# twice over (n_obs_steps=2) and once per camera. That is four large images per
# sample and no augmentation. resize_shape brings them to 240x320 and crop_ratio
# takes a 95% random crop, which is the lerobot recipe for real-robot data.
#
#   bash tools/diffusion_probe.sh            # writes $BASE/logs/diffprobe.log
set -uo pipefail
BASE=/home/padiac/lerobot-train
cd "$BASE" || exit 1

"$BASE/.venv/bin/lerobot-train" \
    --policy.type=diffusion \
    --dataset.repo_id=local/so101_v3_trim \
    --dataset.root="$BASE/data/so101_v3_trim" \
    --policy.device=cuda --policy.push_to_hub=false \
    --policy.resize_shape="${RESIZE:-[240,320]}" --policy.crop_ratio=0.95 \
    --steps=${STEPS:-200} --save_freq=100000 --log_freq=50 \
    --batch_size="${BATCH:-8}" --num_workers=4 \
    --output_dir=$BASE/outputs/diffprobe_$(date +%H%M%S) > ${LOG:-$BASE/logs/diffprobe.log} 2>&1
echo "rc=$?"
tail -3 ${LOG:-$BASE/logs/diffprobe.log}
