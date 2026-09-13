r"""A page of buttons that run fixed-length trials against a robot kept on standby.

The version before this one started a whole rollout per press. That gave perfect
trial boundaries -- the process ended, so the trial ended -- and cost about eighty
seconds of loading the checkpoint, opening two cameras, starting rerun and homing,
every single time. Two trials in a row were unusable.

So: one long-lived rollout holds the arm, the cameras, the model and the rerun
window for the whole sitting. With no instruction it produces no action at all
(the idle guard in live_task.py), so standby really is standby -- the arm does not
move until asked. A press writes the instruction, the arm works for a fixed number
of seconds, and then the panel clears it and the arm returns to its start pose.

The fixed length no longer comes from a process exiting. Every button is greyed
out from the moment one is pressed until the arm is back on standby, which is all
that discipline ever needed to be.

    .\.venv-win\Scripts\python.exe panel.py --policy policies\smolvla_color6only_20k

Add --stub to click through the page without a robot: a small script stands in for
the rollout and moves through the same states.
"""

import argparse
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Which file carries the instruction. It is a setting, not a constant, because a
# test that writes the live one is a test that can move the real arm: while a
# session was running, a test run wrote instructions into the same task.txt the
# rollout was reading, and read back the arm's real state as if it were its own.
TASK_FILE = HERE / "task.txt"
STATUS_FILE = HERE / "task.status"

# Sentence shapes taken verbatim from the recorded task strings -- the out-of-bowl
# episodes say "cube", the into-bowl ones say "block".
OUT_OF_BOWL = "Pick the {} cube out of the bowl and put it on the table."
INTO_BOWL = "Pick the {} block and put it in the black bowl."

COLOURS = [
    ("red", "#e8453c"),
    ("blue", "#3b78e7"),
    ("yellow", "#f0c000"),
    ("green", "#2ea44f"),
    ("purple", "#8b5cf6"),
    ("pink", "#ec7fa9"),
]

# lerobot logs this once the policy is driving the arm. Everything before it is
# the eighty seconds of setup this design exists to pay only once.
LOOP_MARKER = "control loop started"

# lerobot warns whenever the control loop falls behind, and it prints the rate it
# actually achieved. That number decides whether a trial means anything: the
# policy was trained at 30 fps, a demonstration is about 470 control steps, and at
# 1.9 Hz a 30-second trial executes 57 of them -- an eighth of the reach. A model
# judged under that is not being judged at all, so the rate belongs on the page.
RATE_RE = re.compile(r"running slower \(([0-9.]+) Hz\)")
DEMO_STEPS = 470

# The session is given a duration far longer than any sitting; it ends by being
# closed, not by running out.
SESSION_HOURS = 10

_session = {"proc": None, "log": None, "spawned": 0.0}
# phase: ready -> sent -> running -> ending -> ready
_trial = {"phase": "ready", "label": "", "task": "", "sent": 0.0,
          "running_since": 0.0, "worked": 0.0, "last": ""}
_lock = threading.Lock()

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>SO-101</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#111318; color:#e8eaed; min-height:100vh;
         font:15px/1.5 ui-sans-serif,system-ui,"Segoe UI",sans-serif;
         display:flex; flex-direction:column; align-items:center; padding:26px 16px 40px; }
  h1 { font-size:15px; font-weight:600; margin:0 0 4px; letter-spacing:.02em; }
  p.sub { margin:0 0 20px; color:#9aa0a6; font-size:13px; }
  .wrap { width:min(520px,100%); }
  .head { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:8px; }
  .head div { text-align:center; font-size:12px; color:#9aa0a6; letter-spacing:.06em;
              text-transform:uppercase; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  button { border:0; border-radius:11px; padding:17px 10px; font-size:16px; font-weight:600;
           color:#0b0d10; cursor:pointer;
           transition:transform .06s, filter .15s, opacity .15s; }
  button:hover:not(:disabled) { filter:brightness(1.13); }
  button:active:not(:disabled) { transform:scale(.97); }
  button.ghost { background:transparent; color:#e8eaed; }
  button:disabled { cursor:not-allowed; opacity:.26; }
  button.active { opacity:1; outline:2px solid #e8eaed; outline-offset:2px; }
  #now { margin-top:18px; border-radius:10px; padding:14px 16px; font-size:13px;
         background:#191c22; color:#9aa0a6; box-shadow:inset 0 0 0 1px #23262d; }
  #now b { color:#e8eaed; font-weight:600; }
  #now.busy { color:#f0c000; box-shadow:inset 0 0 0 1px #4a3f10; }
  #now.bad { color:#f08a7a; box-shadow:inset 0 0 0 1px #4a1f18; }
  #bar { height:3px; margin-top:10px; border-radius:2px; background:#23262d; overflow:hidden; }
  #fill { height:100%; width:0; background:#f0c000; transition:width .3s linear; }
</style>
<h1>SO-101</h1>
<p class="sub">the arm stays on standby; one press runs one __DUR__-second trial</p>
<div class="wrap">
  <div class="head"><div>take out of bowl</div><div>put into bowl</div></div>
  <div class="grid" id="grid"></div>
  <div id="now">starting the session ...</div>
  <div id="bar"><div id="fill"></div></div>
</div>
<script>
const COLOURS = __COLOURS__;
const OUT = __OUT__, IN = __IN__, DUR = __DUR__;
const grid = document.getElementById('grid');
const now = document.getElementById('now');
const fill = document.getElementById('fill');
const buttons = [];

function mk(colour, css, task, ghost) {
  const b = document.createElement('button');
  b.textContent = colour;
  b.dataset.label = colour + (ghost ? ' -> bowl' : ' out');
  if (ghost) { b.classList.add('ghost'); b.style.boxShadow = 'inset 0 0 0 2px ' + css; }
  else b.style.background = css;
  b.onclick = () => start(task, b.dataset.label);
  buttons.push(b);
  return b;
}
for (const [colour, css] of COLOURS) {
  grid.appendChild(mk(colour, css, OUT.replace('{}', colour), false));
  grid.appendChild(mk(colour, css, IN.replace('{}', colour), true));
}

async function start(task, label) {
  await fetch('/run', {method:'POST', body: JSON.stringify({task, label})});
  poll();
}

// The page never decides for itself whether the arm is free; it asks, and the
// server answers from one function. When those were two different questions, the
// page invited presses that the server then silently refused.
async function poll() {
  let s;
  try { s = await (await fetch('/status')).json(); }
  catch (e) { now.textContent = 'panel lost its own server'; return; }

  // Greyed out from the press until the arm is back on standby. That, and
  // nothing else, is what keeps every trial the same length.
  const free = s.state === 'ready';
  for (const b of buttons) {
    b.disabled = !free;
    b.classList.toggle('active', !free && b.dataset.label === s.label);
  }
  now.className = s.state === 'offline' ? 'bad' : (free ? '' : 'busy');

  if (s.state === 'booting') {
    now.innerHTML = 'starting the session: loading the policy, opening the cameras ' +
                    'and homing &nbsp;<b>' + Math.round(s.elapsed) + 's</b>' +
                    '<br>this happens once for the whole sitting, not once per trial';
    fill.style.width = '0';
  } else if (s.state === 'offline') {
    now.innerHTML = 'the session is not running. ' + (s.note || '') +
                    '<br>stop the panel and start it again';
    fill.style.width = '0';
  } else if (s.state === 'homing') {
    now.innerHTML = '<b>' + s.label + '</b> &nbsp;sent; the arm goes to its start pose ' +
                    'first';
    fill.style.width = '0';
  } else if (s.state === 'running') {
    now.innerHTML = '<b>' + s.label + '</b> &nbsp;' +
                    Math.max(0, Math.ceil(s.remaining)) + 's left';
    fill.style.width = (100 * (1 - s.remaining / DUR)) + '%';
  } else if (s.state === 'ending') {
    now.innerHTML = '<b>' + s.label + '</b> &nbsp;time is up, going back to the start ' +
                    'pose ...';
    fill.style.width = '100%';
  } else {
    fill.style.width = '0';
    let head = s.last ? 'ready &nbsp;&nbsp;<span style="opacity:.7">last: ' +
                        s.last + '</span>'
                      : 'ready -- the arm is on standby';
    if (s.covers !== null && s.covers !== undefined) {
      const pct = Math.round(s.covers * 100);
      const bad = s.covers < 1.0;
      head += '<br><span style="color:' + (bad ? '#f0c000' : '#9aa0a6') +
              '">control loop ' + s.hz.toFixed(1) + ' Hz &nbsp;a trial covers ' +
              pct + '% of one demonstration' +
              (bad ? ' -- too short to finish the motion' : '') + '</span>';
    }
    now.innerHTML = head;
  }
}
poll();
setInterval(poll, 300);
</script>
"""


def _read_log() -> str:
    """Read the session log whatever encoding it landed in.

    PowerShell 5.1's Tee-Object writes UTF-16LE with a BOM, not UTF-8, so reading
    it as UTF-8 yields bytes that match nothing and the marker below is never
    found. The stub writes plain UTF-8, so decide from the BOM rather than from
    which one wrote it.
    """
    log = _session["log"]
    if log is None:
        return ""
    try:
        raw = log.read_bytes()
    except OSError:
        return ""
    if raw[:2] in (bytes([0xFF, 0xFE]), bytes([0xFE, 0xFF])):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _arm() -> dict:
    """What live_task.py says the arm is doing: idle, homing, running or stuck."""
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"mode": "", "task": ""}


def _send(text: str) -> None:
    """Write the instruction the rollout follows. Empty means stop and go home."""
    # No BOM: the policy would otherwise be asked to follow a string starting with
    # U+FEFF, which is not one it was trained on.
    TASK_FILE.write_text(text, encoding="utf-8")


def start_session(cfg) -> None:
    """Bring up the one rollout that holds the arm for the whole sitting."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = HERE / "logs" / f"session_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _send("")                                   # standby before anything starts

    env = dict(os.environ, LIVE_TASK_FILE=str(TASK_FILE))
    if cfg.stub:
        cmd = [sys.executable, str(HERE / "tools" / "fake_session.py")]
    else:
        # Its own console window. Piped into a file, the rollout loaded the policy,
        # started rerun and then sat for an hour on 0.8 seconds of CPU: it wants a
        # console, and rerun wants a desktop session it can draw into.
        # Tee-Object keeps the log too, and *>&1 folds in Write-Host, which in
        # PowerShell 5.1 goes to the information stream and would miss the file.
        nodisp = " -NoDisplay" if cfg.no_display else ""
        # Only needed when the checkpoint's training copy has no local twin, as
        # for the pooled datasets. robot.ps1 checks whatever it is given against
        # the checkpoint's own statistics either way.
        homeds = f" -HomeDataset '{cfg.home_dataset}'" if cfg.home_dataset else ""
        ps = (f"& {{ .\\robot.ps1 eval -Policy '{cfg.policy}' -Live{nodisp}{homeds} "
              f"-Duration {SESSION_HOURS * 3600} }} *>&1 | "
              f"Tee-Object -FilePath '{log}'")
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps]

    if cfg.stub:
        # The stub has no console to write to, so its log is redirected here.
        # The real one writes its own via Tee-Object.
        kw = {"stdout": open(log, "w", encoding="utf-8", errors="replace"),
              "stderr": subprocess.STDOUT, "text": True, "bufsize": 1}
    else:
        kw = {"creationflags": subprocess.CREATE_NEW_CONSOLE}
    proc = subprocess.Popen(cmd, cwd=str(HERE), env=env, **kw)
    _session.update(proc=proc, log=log, spawned=time.time())
    print(f"session starting, log {log.name}", flush=True)


def _rate() -> float:
    """The control rate lerobot last reported, or 0 if it has not complained."""
    hits = RATE_RE.findall(_read_log())
    return float(hits[-1]) if hits else 0.0


def _ready() -> bool:
    """Has the policy taken over? Until then the arm is still being set up."""
    return LOOP_MARKER in _read_log()


def _tick(cfg) -> None:
    """Move the trial along. One place decides; nothing else writes the phase."""
    with _lock:
        ph = _trial["phase"]
        if ph == "ready":
            return
        mode = _arm().get("mode", "")

        if ph == "sent":
            # live_task homes to the training start pose before every instruction,
            # so the clock starts when the arm actually begins working. Counting
            # the homing against it would give each instruction a different amount
            # of real working time.
            if mode == "running":
                _trial.update(phase="running", running_since=time.time())
            elif mode == "stuck":
                # live_task could not get the arm home and is deliberately holding
                # still rather than starting from a pose the policy never saw.
                # That is an answer, not a hang: say so and hand the buttons back.
                _send("")
                _trial.update(phase="ready",
                              last=f"{_trial['label']} -- could not reach the start "
                                   "pose, see the session window")
                print(f"  {_trial['last']}", flush=True)
            elif time.time() - _trial["sent"] > 120:
                _send("")
                _trial.update(phase="ready",
                              last=f"{_trial['label']} -- never started, see the window")
                print(f"  {_trial['last']}", flush=True)
        elif ph == "running":
            if time.time() - _trial["running_since"] >= cfg.duration:
                _send("")                       # cleared; the arm goes home
                # How long it actually worked, measured to the moment the
                # instruction was cleared. Measuring to the moment it finished
                # going home added the homing on top and reported a 30-second
                # trial as 36 seconds.
                _trial.update(phase="ending",
                              worked=time.time() - _trial["running_since"])
        elif ph == "ending":
            if mode in ("idle", ""):
                _trial.update(phase="ready",
                              last=f"{_trial['label']}, {_trial['worked']:.0f}s")
                print(f"  done  {_trial['last']}", flush=True)
            elif mode == "stuck":
                _trial.update(phase="ready",
                              last=f"{_trial['label']}, {_trial['worked']:.0f}s "
                                   "-- did not get all the way home")
                print(f"  done  {_trial['last']}", flush=True)
            elif time.time() - _trial["running_since"] > cfg.duration + 120:
                _send("")
                _trial.update(phase="ready",
                              last=f"{_trial['label']} -- never reported going home")
                print(f"  {_trial['last']}", flush=True)


def _ticker(cfg) -> None:
    while True:
        try:
            _tick(cfg)
        except Exception as e:                  # never let the loop die silently
            print(f"  ticker error: {type(e).__name__}: {e}", flush=True)
        time.sleep(0.2)


def _state_unlocked(cfg) -> dict:
    """The single answer to "can a button be pressed right now".

    It has to be the only one. When the page asked one question and the button
    handler asked a different one, the two disagreed exactly when it mattered and
    every press was refused without a word.
    """
    proc = _session["proc"]
    if proc is None or proc.poll() is not None:
        rc = None if proc is None else proc.returncode
        return {"state": "offline",
                "note": "" if rc is None else f"it exited with code {rc}."}
    if not _ready():
        return {"state": "booting", "elapsed": time.time() - _session["spawned"]}

    ph = _trial["phase"]
    if ph == "sent":
        return {"state": "homing", "label": _trial["label"]}
    if ph == "running":
        left = cfg.duration - (time.time() - _trial["running_since"])
        return {"state": "running", "label": _trial["label"],
                "remaining": max(0.0, left)}
    if ph == "ending":
        return {"state": "ending", "label": _trial["label"]}
    hz = _rate()
    return {"state": "ready", "last": _trial["last"], "hz": hz,
            # How much of one demonstration fits in a trial at this rate. Below
            # about 1.0 the arm cannot finish what it was shown, whatever it
            # learned.
            "covers": (hz * cfg.duration / DEMO_STEPS) if hz else None}


def _state(cfg) -> dict:
    with _lock:
        return _state_unlocked(cfg)


class Handler(http.server.BaseHTTPRequestHandler):
    cfg = None

    def log_message(self, *a):
        pass                                    # the page is the log

    def _json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/status"):
            self._json(_state(self.cfg))
            return
        body = (
            PAGE.replace("__COLOURS__", json.dumps(COLOURS))
            .replace("__OUT__", json.dumps(OUT_OF_BOWL))
            .replace("__IN__", json.dumps(INTO_BOWL))
            .replace("__DUR__", str(self.cfg.duration))
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        with _lock:
            if _state_unlocked(self.cfg)["state"] != "ready":
                self._json({"busy": True})
                return
            _trial.update(phase="sent", label=req.get("label", "?"),
                          task=req.get("task", ""), sent=time.time(),
                          running_since=0.0)
            _send(_trial["task"])
            print(f"  start {_trial['label']}: {_trial['task']}", flush=True)
        self._json({"started": True})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=r"policies\smolvla_color6only_20k")
    ap.add_argument("--home-dataset", default="",
                    help="where to read the start pose from, when the "
                         "checkpoint's training copy has no local twin")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--stub", action="store_true",
                    help="a fake session, to check the page without a robot")
    ap.add_argument("--task-file", type=Path, default=HERE / "task.txt",
                    help="the file the rollout reads its instruction from; "
                         "point tests somewhere else so they cannot drive the arm")
    ap.add_argument("--no-display", action="store_true",
                    help="run without the rerun window; it costs about three "
                         "quarters of the control rate in a long session")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    global TASK_FILE, STATUS_FILE
    TASK_FILE = args.task_file.resolve()
    STATUS_FILE = TASK_FILE.with_suffix(".status")

    if not args.stub and not (HERE / args.policy / "config.json").exists():
        print(f"no policy at {HERE / args.policy}")
        return 1

    Handler.cfg = args
    # NOT allow_reuse_address: on Windows that lets a second panel bind a port
    # another panel already serves. Presses then land on whichever socket wins
    # while the other holds stale state and its buttons do nothing.
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        print(f"port {args.port} is already in use ({e.strerror}).")
        print("A panel is probably already running. Use that one, or --port.")
        return 1

    start_session(args)
    threading.Thread(target=_ticker, args=(args,), daemon=True).start()

    with httpd:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"policy   {args.policy}" + ("   [stub]" if args.stub else ""))
        print(f"trial    {args.duration}s, arm on standby in between")
        print(f"open {url}   (ctrl-c to stop)")
        if not args.no_open:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
        finally:
            _send("")                           # leave the arm on standby
            p = _session["proc"]
            if p is not None and p.poll() is None:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                               capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
