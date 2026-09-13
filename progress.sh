#!/usr/bin/env bash
# Where is the training run up to? Reads only what is on disk, so it works from
# any shell and survives the session that launched the run.
#
# Progress is derived from the checkpoint directory rather than a log file:
# lerobot names each checkpoint by the global step, so the newest one plus the
# gap between the last two gives both the position and the rate.
BASE=/home/padiac/lerobot-train
OUT=${1:-$(ls -dt "$BASE"/outputs/*/ 2>/dev/null | head -1)}
OUT=${OUT%/}
CK="$OUT/checkpoints"

echo "run     $(basename "$OUT")"

if ! pgrep -f "bin/lerobot-train" > /dev/null; then
    echo "status  NOT RUNNING"
else
    echo "status  running (pid $(pgrep -f 'bin/lerobot-train' | head -1))"
fi

test -d "$CK" || { echo "no checkpoints yet"; exit 0; }

mapfile -t CKS < <(ls -d "$CK"/[0-9]* 2>/dev/null | sort)
n=${#CKS[@]}
[ "$n" -gt 0 ] || { echo "no checkpoints yet"; exit 0; }

last=${CKS[$((n-1))]}
step=$((10#$(basename "$last")))
total=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['steps'])" \
        "$last/pretrained_model/train_config.json" 2>/dev/null || echo 0)

printf "step    %d" "$step"
[ "$total" -gt 0 ] && printf " / %d  (%.1f%%)" "$total" "$(echo "$step $total" | awk '{print 100*$1/$2}')"
echo

# Rate, measured only over what THIS process has done. Using the wall-clock gap
# between the last two checkpoints instead would fold in any downtime between a
# crash and the restart -- after the 2026-09-01 crash that reported 1.95 step/s
# for a run actually doing 6.4, and an ETA five hours too pessimistic.
PID=$(pgrep -f "bin/lerobot-train" | head -1)
if [ -n "$PID" ] && [ "$n" -ge 2 ]; then
    started=$(stat -c %Y "/proc/$PID")
    prev=${CKS[$((n-2))]}
    if [ "$(stat -c %Y "$last")" -gt "$started" ]; then
        # Reference point is whichever is later: the previous checkpoint, or the
        # moment this process resumed from it.
        ref=$(stat -c %Y "$prev")
        [ "$ref" -lt "$started" ] && ref=$started
        dt=$(( $(stat -c %Y "$last") - ref ))
        ds=$(( step - 10#$(basename "$prev") ))
        if [ "$dt" -gt 0 ] && [ "$ds" -gt 0 ]; then
            rate=$(echo "$ds $dt" | awk '{printf "%.2f", $1/$2}')
            echo "rate    $rate step/s  (this process, last $ds steps)"
            if [ "$total" -gt "$step" ]; then
                eta=$(echo "$total $step $rate" | awk '{printf "%d", ($1-$2)/$3}')
                echo "left    $((eta/3600))h $(((eta%3600)/60))m   done around $(date -d "+$eta seconds" '+%H:%M')"
            fi
        fi
    fi
fi
echo "saved   $(date -r "$last" '+%H:%M:%S')  ($(ls -d "$CK"/[0-9]* | wc -l) checkpoints)"
echo
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader \
    | awk -F', ' '{printf "gpu     %s / %s   util %s   %s\n", $1, $2, $3, $4}'
free -m | awk '/^Mem:/{printf "wsl ram %d / %d MB used\n", $3, $2}'
