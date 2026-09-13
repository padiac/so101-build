B=/home/padiac/lerobot-train
echo "--- 最新日志 ---"
tail -8 $(ls -t $B/logs/color6_armed_*.log | head -1)
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- 进程 ---"
pgrep -af "pipeline_color6|lerobot-train|lerobot-edit|trim_dataset" | head -3
