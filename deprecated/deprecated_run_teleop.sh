#!/usr/bin/env bash
# 带日志的遥操作启动器。
#
# 会做三件事：
#   1. 跑之前抓一次 health 快照（电压/温度/错误标志）
#   2. 跑 lerobot-teleoperate，全部输出同时进终端和日志文件
#   3. 退出后再抓一次 health 快照 —— 崩溃时这一份最有用
#
# 用法：
#   ./run_teleop.sh                 # 默认 max_relative_target=15, fps=30
#   ./run_teleop.sh 25 30           # 自定义 限速 fps
#
# 日志在 logs/ 下，文件名带时间戳。

set -u

MRT="${1:-15}"
FPS="${2:-30}"

FOLLOWER_PORT=/dev/ttyACM1
LEADER_PORT=/dev/ttyACM0

cd "$(dirname "$0")" || exit 1
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/teleop_${STAMP}.log"

{
  echo "=========================================================="
  echo "run_teleop.sh  $(date)"
  echo "max_relative_target=${MRT}  fps=${FPS}"
  echo "follower=${FOLLOWER_PORT}  leader=${LEADER_PORT}"
  echo "=========================================================="
  echo
  echo "########## HEALTH BEFORE ##########"
} | tee "$LOG"

python health.py --leader-port "$LEADER_PORT" --follower-port "$FOLLOWER_PORT" 2>&1 | tee -a "$LOG"

{
  echo
  echo "########## TELEOP START ##########"
} | tee -a "$LOG"

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=my_follower \
  --robot.max_relative_target="$MRT" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=my_leader \
  --fps="$FPS" 2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}

{
  echo
  echo "########## TELEOP EXIT rc=${RC} ##########"
  echo
  echo "########## HEALTH AFTER ##########"
} | tee -a "$LOG"

python health.py --leader-port "$LEADER_PORT" --follower-port "$FOLLOWER_PORT" 2>&1 | tee -a "$LOG"

echo
echo "日志已保存: $LOG"
