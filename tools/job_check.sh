#!/usr/bin/env bash
# One look at the current training job: alive, moving, finished, or dead.
#
# Why this exists. A long run that dies is only half the problem; the other half
# is nobody noticing. On 2026-10-15 the 02:00 run died at step 20000 on a full
# disk and nothing looked at it until the person asked eight hours later. Every
# long job now gets three scheduled looks spread over its expected length, and a
# restart gets three new ones. This is what each look runs.
#
# "Moving" is judged from the newest write anywhere the job writes, not from the
# log alone: the trim step pipes its output through tail, so the log is silent
# for the hour or two it takes while the trimmed videos grow underneath it.
#
#   wsl -e bash /mnt/e/Repo/so101-build/tools/job_check.sh
#
# Exit status: 0 running, 10 finished, 20 failed / dead / stalled.
BASE=/home/padiac/lerobot-train
STALL_MIN=${STALL_MIN:-15}

log=$(readlink -f "${LOG:-$BASE/logs/current.log}")
[ -f "$log" ] || { echo "VERDICT: NO-JOB (no current.log)"; exit 20; }
now=$(date +%s)

# Newest write by the job: its log, the prepared data, or the run's outputs.
newest=$(stat -c %Y "$log")
for d in "$BASE/data" "$BASE/outputs"; do
    t=$(find "$d" -maxdepth 4 -newer "$log" -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    [ -n "$t" ] && [ "${t%.*}" -gt "$newest" ] && newest=${t%.*}
done
quiet_min=$(( (now - newest) / 60 ))

alive=$(pgrep -f "$BASE/run/|bin/lerobot-train|trim_dataset.py|lerobot-edit-dataset" | head -1)
step=$(tail -c 20000 "$log" | tr '\r' '\n' | grep -aoE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+' | tail -1)
out=$(grep -a 'output: ' "$log" | tail -1 | sed 's/.*output: //')

echo "log:      $log"
echo "last say: $(grep -a '^\[' "$log" | tail -1)"
echo "progress: ${step:-n/a (not training yet)}"
echo "quiet:    ${quiet_min} min since the job last wrote anything"
echo "process:  $([ -n "$alive" ] && echo "alive (pid $alive)" || echo none)"
echo "disk:     $(df -BG --output=avail "$BASE" | tail -1 | tr -d ' ') free"

if grep -aq '\] done: ' "$log"; then
    echo "VERDICT: DONE -- $(grep -a 'publishing -> ' "$log" | tail -1 | sed 's/.*-> //')"
    exit 10
fi
if [ -z "$alive" ]; then
    echo "VERDICT: DEAD -- $(grep -aE '!!!|Error|error:' "$log" | tail -2 | tr '\n' ' ')"
    [ -n "$out" ] && echo "resume:   RESUME_OUT=$out bash /mnt/e/Repo/so101-build/tools/train_emptybowl.sh"
    exit 20
fi
if [ "$quiet_min" -ge "$STALL_MIN" ]; then
    echo "VERDICT: STALLED -- alive but silent for ${quiet_min} min"
    exit 20
fi
echo "VERDICT: RUNNING"
exit 0
