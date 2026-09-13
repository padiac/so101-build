r"""Check the panel's standby loop end to end, without a robot.

Starts a real panel against a fake session that moves through the same states a
live rollout does, then drives it over HTTP exactly as the page does. Every
failure covered here was found on the bench first, at the cost of an evening
each, and none of them needed the arm.

    .\.venv-win\Scripts\python.exe tools\policy\test_panel.py
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PORT = 8814
DUR = 4
RED = "Pick the red cube out of the bowl and put it on the table."
BLUE = "Pick the blue cube out of the bowl and put it on the table."


def get():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/status", timeout=5) as r:
        return json.load(r)


def press(task, label):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/run",
        data=json.dumps({"task": task, "label": label}).encode())
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def wait_for(state, limit=40):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        s = get()
        if s["state"] == state:
            return s, time.perf_counter() - t0
        time.sleep(0.1)
    raise AssertionError(f"never reached {state!r}; stuck at {get()}")


# Its own instruction file, never the live one: a test that writes task.txt is a
# test that can move the real arm if a session happens to be running.
TASK = ROOT / "logs" / "test_panel_task.txt"
TASK.parent.mkdir(parents=True, exist_ok=True)

panel = subprocess.Popen(
    [sys.executable, str(ROOT / "panel.py"), "--stub", "--duration", str(DUR),
     "--port", str(PORT), "--no-open", "--task-file", str(TASK)],
    cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    for _ in range(60):
        try:
            get()
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)

    print("1. the session boots once, and the buttons stay dead until it is up")
    s = get()
    assert s["state"] == "booting", s
    assert press(RED, "red out").get("busy"), "a press was accepted while booting"
    s, took = wait_for("ready")
    print(f"   ready after {took:.1f}s, arm on standby  OK")

    print("2. a press homes first, then runs, then goes back to standby")
    assert press(RED, "red out").get("started")
    _, t_home = wait_for("homing", 5)
    s, _ = wait_for("running", 15)
    assert s["label"] == "red out", s
    # The clock starts when the arm begins working, not when the button was
    # pressed: otherwise the homing eats a different slice of every trial.
    assert s["remaining"] <= DUR + 0.1, s
    print(f"   homing, then running with {s['remaining']:.1f}s left  OK")

    print("3. every button is dead for the whole trial")
    assert press(BLUE, "blue out").get("busy"), "a press was accepted mid-trial"
    wait_for("ending", 20)
    assert press(BLUE, "blue out").get("busy"), "a press was accepted while going home"
    print("   refused while running and while going home  OK")

    print("4. it comes back by itself, and reports what the trial was")
    s, _ = wait_for("ready", 20)
    assert s["last"].startswith("red out"), s
    secs = int(s["last"].rsplit(",", 1)[1].strip().rstrip("s"))
    assert abs(secs - DUR) <= 1, f"reported {secs}s for a {DUR}s trial"
    print(f"   ready again, last: {s['last']}  OK")

    print("5. the next press runs at once -- no reboot, no eighty seconds")
    t0 = time.perf_counter()
    assert press(BLUE, "blue out").get("started")
    s, _ = wait_for("running", 15)
    assert s["label"] == "blue out", s
    print(f"   second trial running {time.perf_counter() - t0:.1f}s after the press  OK")

    print("6. and the session was never restarted for it")
    assert get()["state"] != "booting"
    print("   still the same session  OK")

finally:
    # Kill the fake session before the panel: it holds the log open, and it is a
    # grandchild, so terminating the panel alone leaves it running.
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*fake_session*' } | "
         "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
        capture_output=True)
    panel.terminate()
    try:
        panel.wait(timeout=10)
    except subprocess.TimeoutExpired:
        panel.kill()
    for _ in range(20):                     # the handle takes a moment to drop
        left = [f for f in ROOT.glob("logs/session_*.log")]
        for f in left:
            try:
                f.unlink()
            except OSError:
                pass
        if not [f for f in ROOT.glob("logs/session_*.log")]:
            break
        time.sleep(0.2)
    TASK.unlink(missing_ok=True)
    TASK.with_suffix(".status").unlink(missing_ok=True)

print("\nALL OK")
