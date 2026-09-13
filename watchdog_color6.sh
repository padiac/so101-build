#!/usr/bin/env bash
# Keep the colour-6 SmolVLA run alive while nobody is watching.
#
# Called from a Windows scheduled task every few hours, because that is the only
# layer that survives WSL shutting itself down or the machine rebooting -- an
# in-WSL loop does not. It is idempotent and cheap: if the run is healthy or
# already finished it does nothing.
#
# Restart policy: only restart if the checkpoint step has advanced since the last
# restart. A run that dies twice without progressing is a real fault, and looping
# on it just burns two days and produces ten identical tracebacks.
set -uo pipefail

BASE=/home/padiac/lerobot-train
WIN=/mnt/e/Repo/so101-build
DS=so101_mix1
TARGET=${STEPS:-20000}
STATE=$BASE/logs/watchdog_${DS}.state
LOG=$BASE/logs/watchdog_${DS}.log

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
# DRY=1 prints the restart it would perform instead of performing it.
RUNNER=""
if [ "${DRY:-0}" = "1" ]; then RUNNER="echo [DRY] would run:"; STATE=$BASE/logs/watchdog_${DS}.dry.state; fi

latest_out() { ls -dt "$BASE"/outputs/smolvla_${DS}_* 2>/dev/null | head -1; }

last_step() {
    local o ck l
    o=$(latest_out) || return 1
    [ -n "$o" ] || return 1
    ck="$o/checkpoints"
    [ -d "$ck" ] || return 1
    l=$(ls -d "$ck"/[0-9]* 2>/dev/null | sort | tail -1) || return 1
    [ -n "$l" ] || return 1
    echo $((10#$(basename "$l")))
}

# --- 1. finished? ---
step=$(last_step) || step=0
if [ "$step" -ge "$TARGET" ]; then
    say "done: step $step >= $TARGET, nothing to do"
    exit 0
fi

# --- 2. healthy? (the armed job counts, even while it is still sleeping) ---
if pgrep -f "pipeline_color6.sh" > /dev/null || pgrep -f "bin/lerobot-train" > /dev/null \
   || pgrep -f "trim_dataset.py" > /dev/null; then
    say "healthy: running, step $step/$TARGET"
    exit 0
fi

# --- 3. it is down. Two ways to loop forever, so guard both ---
#
#   a) it crashes mid-training and comes back to the same checkpoint
#   b) it fails at startup and never writes a checkpoint at all
#
# The first version only guarded (a) -- it required a previous step > 0. A config
# error (wrong camera names, missing checkpoint) fails in the first second every
# single time, leaves step at 0 forever, and the guard never fired. So count
# fruitless restarts as well as compare steps.
prev=0
fails=0
if [ -f "$STATE" ]; then
    prev=$(awk 'NR==1{print $1+0}' "$STATE" 2>/dev/null || echo 0)
    fails=$(awk 'NR==2{print $1+0}' "$STATE" 2>/dev/null || echo 0)
fi

if [ "$step" -gt "$prev" ]; then
    fails=0                       # real progress since the last restart
else
    fails=$((fails + 1))
fi

if [ "$fails" -ge 3 ]; then
    say "DOWN at step $step after $fails restarts with no progress."
    say "That is a deterministic fault, not a transient one -- restarting again"
    say "would just produce the same failure every $((4)) hours until someone looks."
    say "Stopping. Check the newest log under $BASE/logs/."
    exit 2
fi

printf '%s
%s
' "$step" "$fails" > "$STATE"
say "restart attempt $fails (step $step, previous best $prev)"

# --- 4. restart: resume if there is a checkpoint, otherwise start from scratch ---
o=$(latest_out)
if [ "$step" -gt 0 ] && [ -d "$o/checkpoints/last/pretrained_model" ]; then
    say "DOWN at step $step -- resuming from checkpoint"
    $RUNNER setsid nohup "$BASE/.venv/bin/lerobot-train" --resume=true \
        --config_path="$o/checkpoints/last/pretrained_model" \
        >> "$BASE/logs/resume_${DS}_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
else
    say "DOWN with no checkpoint -- starting the pipeline from the beginning"
    $RUNNER setsid nohup env DELAY=0 STEPS="$TARGET" BATCH=16 bash "$WIN/pipeline_color6.sh" \
        >> "$BASE/logs/restart_${DS}_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
fi
sleep 3
say "restarted; pids: $(pgrep -f 'pipeline_color6.sh|bin/lerobot-train' | tr '\n' ' ')"
