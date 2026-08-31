#!/usr/bin/env bash
# ACT 训练 —— 在 WSL 里跑，由 Windows 任务计划程序在 3am 调用。
#
# 手动跑：
#   wsl -d Ubuntu -- bash /mnt/e/Repo/so101-build/train_act.sh
#
# 参数依据（2026-08-26 烟雾测试实测）：
#   6 step/s，updt_s 0.167，data_s 0.003 -> 瓶颈在 GPU 不在数据加载
#   steps=100000 -> 约 56 个 epoch（14332 帧 / batch 8）-> 约 4.6 小时
#   batch_size=8 是 ACT 预设调好的值，改它要连学习率一起调，第一轮不动
set -euo pipefail

V=/home/padiac/lerobot-train/.venv/bin
BASE=/home/padiac/lerobot-train
# STEPS 可用环境变量覆盖，用于用少量步数验证调用路径：
#   wsl -d Ubuntu -- bash -c 'STEPS=20 bash /mnt/e/Repo/so101-build/train_act.sh'
STEPS=${STEPS:-100000}
# 数据集可覆盖，便于对比裁剪前后
DATA=${DATA:-so101_pickplace}

# chunk_size / n_action_steps: lerobot 默认 100/100，即一次规划 3.3 秒并全部开环执行。
# 实测那样会乱走，而调小 n_action_steps 又卡在轨迹的慢起步段（README #28）。
# Trelis 在 SO-101 上用的是 50/15 —— 预测 1.67 秒、执行前 0.5 秒，
# 兼顾了动作分块的好处和闭环的响应。chunk_size 烘进模型，必须在训练时定。
CHUNK=${CHUNK:-50}
ASTEPS=${ASTEPS:-15}

# 小而一致的数据集上关掉 VAE：上一轮 kld_loss 收敛到 0.000，
# 说明隐变量根本没被用上，留着只是多一个不起作用的分支。
USE_VAE=${USE_VAE:-false}

BATCH=${BATCH:-8}
LR=${LR:-1e-5}

# Dataloader buffers scale with the batch, and getting this wrong took the whole
# machine down once. Each sample carries two 640x480x3 float images, about
# 7.4 MB, so one batch of 64 is roughly 470 MB. With 6 workers each prefetching
# 4 batches that is ~11 GB of host RAM -- more than WSL has.
#
#   workers x prefetch x batch x 7.4 MB  must stay well under WSL's memory
#
# So shrink the buffers as the batch grows.
if [ "$BATCH" -ge 48 ]; then
    WORKERS=${WORKERS:-2}; PREFETCH=${PREFETCH:-2}
elif [ "$BATCH" -ge 24 ]; then
    WORKERS=${WORKERS:-3}; PREFETCH=${PREFETCH:-2}
else
    WORKERS=${WORKERS:-6}; PREFETCH=${PREFETCH:-4}
fi
echo "batch $BATCH -> num_workers $WORKERS, prefetch_factor $PREFETCH"
STAMP=$(date +%Y%m%d_%H%M)
OUT=$BASE/outputs/act_${DATA}_$STAMP

mkdir -p "$BASE/logs"
LOG=$BASE/logs/train_${DATA}_$STAMP.log

echo "=== ACT training start $(date) ===" | tee "$LOG"
echo "output: $OUT" | tee -a "$LOG"

# 注意：set -euo pipefail 下管道一失败就立刻退出，
# 后面取 PIPESTATUS 的那行根本执行不到。
# 要把真实退出码打进日志，就得显式关掉再打开。
set +e
"$V/lerobot-train" \
  --dataset.repo_id=local/$DATA \
  --dataset.root=$BASE/data/$DATA \
  --policy.type=act \
  --policy.chunk_size=$CHUNK \
  --policy.n_action_steps=$ASTEPS \
  --policy.use_vae=$USE_VAE \
  --policy.optimizer_lr=$LR \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name="act_${DATA}_$STAMP" \
  --steps=$STEPS \
  --save_freq=10000 \
  --log_freq=500 \
  --batch_size=$BATCH \
  --num_workers=$WORKERS \
  --prefetch_factor=$PREFETCH \
  --wandb.enable=false 2>&1 | tee -a "$LOG"

# tee 的退出码是 0，真实结果在 PIPESTATUS[0]
rc=${PIPESTATUS[0]}
set -e
echo "=== exit $rc  $(date) ===" | tee -a "$LOG"
exit "$rc"
