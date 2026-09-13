#!/usr/bin/env bash
# Watch the next trial and report its timeline from the log it actually writes.
#
# The question this answers is the only one that matters about the panel: after
# one press, when can the next button be pressed? Everything here is read from
# the trial's own log, so the answer does not depend on anyone's belief about
# what the code does.
cd /e/Repo/so101-build || exit 1
BEFORE=$(ls logs/eval_*.log 2>/dev/null | wc -l)

echo "waiting for the next trial ..."
for _ in $(seq 1 240); do            # 20 minutes
    NOW=$(ls logs/eval_*.log 2>/dev/null | wc -l)
    [ "$NOW" -gt "$BEFORE" ] && break
    sleep 5
done
L=$(ls -t logs/eval_*.log 2>/dev/null | head -1)
[ -z "$L" ] && { echo "no trial started"; exit 1; }
echo "trial log: $L"

# Wait for the rollout to report itself done, or give up and say where it stopped.
for _ in $(seq 1 120); do            # 10 minutes
    .venv-win/Scripts/python.exe -c "
import sys; sys.path.insert(0, '.')
import panel; from pathlib import Path
sys.exit(0 if 'Rollout finished' in panel._read_log(Path(r'$L')) else 1)
" && break
    sleep 5
done

.venv-win/Scripts/python.exe -c "
import re, sys; sys.path.insert(0, '.')
import panel
from pathlib import Path
t = panel._read_log(Path(r'$L'))

def when(pat):
    m = re.search(r'(\d\d:\d\d:\d\d)[^\n]*' + pat, t)
    return m.group(1) if m else None

rows = [('policy took over', when('control loop started')),
        ('30s limit reached', when('Duration limit reached')),
        ('arm back at start', when('Returning robot to initial')),
        ('cameras closed', when('SOFollower disconnected')),
        ('ROLLOUT FINISHED', when('Rollout finished'))]
for k, v in rows:
    print('  %-20s %s' % (k, v or 'NOT REACHED'))

done = rows[-1][1]
print()
if done:
    print('The panel goes ready at', done, '-- press the next button from then on.')
else:
    print('It never reported finishing. Last lines of the log:')
    print('\n'.join(t.strip().splitlines()[-8:]))
"
