#!/usr/bin/env bash
# Pre-flight for a long training run. Every check here is one that has already
# cost a run: no disk, no CUDA, wrong venv, or a dataset that will not load.
BASE=/home/padiac/lerobot-train
echo "=== disk (/home) ==="
df -h /home | tail -1
echo
echo "=== memory ==="
free -g | head -2
echo
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "=== venv ==="
test -x "$BASE/.venv/bin/lerobot-train" && echo "lerobot-train present" || echo "MISSING lerobot-train"
"$BASE/.venv/bin/python" - <<'PY'
import torch, lerobot, accelerate
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("lerobot", lerobot.__version__, "accelerate", accelerate.__version__)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
echo
echo "=== existing data in ext4 ==="
du -sh "$BASE"/data/* 2>/dev/null || echo "(none)"
echo
echo "=== source dataset on /mnt/e ==="
du -sh /mnt/e/Repo/so101-build/datasets/so101_v3
