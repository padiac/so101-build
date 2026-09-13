#!/usr/bin/env bash
# Tail the current training log from inside WSL.
#
# Reading it from Windows via \wsl$\... does not work on this machine: the
# distro runs in mirrored networking mode, under which that share frequently
# fails to resolve, and current.log is a symlink besides -- Windows cannot
# follow WSL symlinks over the 9p share even when it does resolve.
B=/home/padiac/lerobot-train
N=${1:-3}
L=$(readlink -f "$B/logs/current.log" 2>/dev/null)
if [ -z "$L" ] || [ ! -f "$L" ]; then
    L=$(ls -t "$B"/logs/*.log 2>/dev/null | head -1)
fi
[ -n "$L" ] || { echo "no training log found under $B/logs"; exit 1; }
echo "log: $L"
echo
# tqdm writes progress with carriage returns, so split on them before tailing.
tr '\r' '\n' < "$L" | grep -vE '^\s*$' | tail -n "$N"
