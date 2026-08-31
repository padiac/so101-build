#!/usr/bin/env bash
# Copy the v3 recording into WSL, trim the dead head, then train.
# Run as one job so a scheduled start needs only a single command.
#
#   wsl -d Ubuntu -- bash /mnt/e/Repo/so101-build/pipeline_v3.sh
set -euo pipefail

WIN=/mnt/e/Repo/so101-build
BASE=/home/padiac/lerobot-train
# Not NAME: that is already set in the Windows environment (the machine name)
# and is inherited through wsl.exe, which silently replaced the default and
# sent the job looking for a dataset called DESKTOP-9J4VMJL.
DS=${DS:-so101_v3}

echo "=== copy $DS into ext4 ==="
# Training reads small files at random; over drvfs (/mnt/e) that is an order of
# magnitude slower than from the WSL filesystem.
rm -rf "$BASE/data/$DS"
cp -r "$WIN/datasets/$DS" "$BASE/data/"

echo "=== trim the still period at the head of each episode ==="
cp "$WIN/trim_dataset.py" "$BASE/trim_dataset.py"
"$BASE/.venv/bin/python" "$BASE/trim_dataset.py" \
  --src "$BASE/data/$DS" --dst "$BASE/data/${DS}_trim" --force

echo "=== train ==="
DATA=${DS}_trim CHUNK=100 ASTEPS=100 USE_VAE=true BATCH=8 LR=1e-5 \
  bash "$WIN/train_act.sh"
