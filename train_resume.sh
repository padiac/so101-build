#!/usr/bin/env bash
# Resume an interrupted lerobot training run, and keep resuming it.
#
# On 2026-09-01 a 100k-step run died at step 48349 with
# "CUDA error: unknown error". The GPU was fine; the Windows host had run out of
# memory (Event 2004, vmmemWSL at 14.6 GB of 15.8 GB) and WSL's paravirtualised
# GPU layer went down first, which PyTorch could only report as an opaque CUDA
# fault. Recovery is cheap -- checkpoints carry training_state -- so the useful
# thing is to make it automatic.
#
#   bash train_resume.sh <output_dir> [max_restarts]
#
# Each attempt resumes from checkpoints/last, so no work beyond the most recent
# checkpoint interval is ever repeated. A restart that fails to advance the step
# counter is treated as a genuine fault rather than a transient one, and the loop
# stops instead of burning the night on a crash loop.
set -uo pipefail

OUT=${1:?usage: train_resume.sh <output_dir> [max_restarts]}
MAX=${2:-10}
V=/home/padiac/lerobot-train/.venv/bin
CKPT="$OUT/checkpoints"

last_step() {
    local l
    l=$(readlink -f "$CKPT/last" 2>/dev/null) || return 1
    basename "$l" | sed 's/^0*//'
}

test -d "$CKPT" || { echo "no checkpoints under $CKPT"; exit 2; }

for ((i = 1; i <= MAX; i++)); do
    before=$(last_step) || { echo "cannot read $CKPT/last"; exit 2; }
    echo "=== attempt $i/$MAX  resuming from step $before  $(date) ==="

    "$V/lerobot-train" --resume=true \
        --config_path="$CKPT/last/pretrained_model"
    rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "=== training finished cleanly $(date) ==="
        exit 0
    fi

    after=$(last_step)
    echo "=== exit $rc at $(date); checkpoint moved $before -> $after ==="

    if [ "$after" = "$before" ]; then
        echo "no progress since the last attempt -- this is not a transient fault."
        echo "stopping so the real cause gets looked at."
        exit "$rc"
    fi

    # Let the host reclaim memory and the GPU settle before trying again.
    sleep 60
done

echo "=== gave up after $MAX attempts ==="
exit 1
