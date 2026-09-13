#!/usr/bin/env bash
# Where the pair run stands: running or not, how far, what has landed.
#
# Reports from the filesystem and the process table, not from hopeful-looking
# text in a log. Three separate "it started" reports on 2026-09-08 turned out to
# be a grep matching a failure line and a model-loading progress bar.
LOG=$(readlink -f /home/padiac/lerobot-train/logs/current.log 2>/dev/null)
WIN=/mnt/e/Repo/so101-build
BASE=/home/padiac/lerobot-train

echo "=== running ==="
if pgrep -f "bin/lerobot-train" > /dev/null; then
    pgrep -af "bin/lerobot-train" | sed "s/ --.*dataset.repo_id=/  /" | cut -c1-120 | head -2
else
    echo "  no training process"
fi
pgrep -f "run/pair_.*.sh" > /dev/null && echo "  wrapper alive" || echo "  wrapper not running"

echo
echo "=== progress ==="
[ -n "$LOG" ] && grep -aoE "\| *[0-9]+/20000 \[[^]]*\]" "$LOG" | tail -1
[ -n "$LOG" ] && grep -aoE "step:[0-9]+ .*loss:[0-9.]+" "$LOG" | tail -1 | cut -c1-100

echo
echo "=== checkpoints written ==="
for d in "$BASE"/outputs/smolvla_so101_v3_* "$BASE"/outputs/smolvla_so101_color6_*; do
    [ -d "$d" ] || continue
    last=$(ls -d "$d"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1)
    printf "  %-40s %s\n" "$(basename "$d")" "${last:+step $(basename "$last")}"
done

echo
echo "=== published to policies/ ==="
for m in smolvla_v3only_20k smolvla_color6only_20k; do
    w=$WIN/policies/$m/model.safetensors
    if [ -f "$w" ]; then
        printf "  %-24s YES  %s MB\n" "$m" "$(( $(stat -c%s "$w") / 1000000 ))"
    else
        printf "  %-24s not yet\n" "$m"
    fi
done

echo
echo "=== pipeline lines ==="
[ -n "$LOG" ] && grep -a "^\[" "$LOG" | tail -8
