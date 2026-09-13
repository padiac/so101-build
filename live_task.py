r"""Change the instruction while the policy is running, one paragraph at a time.

`lerobot-rollout` takes one `--task` string for the whole run, which is fine for a
single-task policy and useless for a multi-task one: to say something else you
have to stop, reload the model and start again -- twenty seconds of dead time per
sentence.

It does not have to be that way. The inference engine keeps the instruction in a
plain attribute and reads it on every inference:

    # lerobot/rollout/inference/sync.py
    self._task = task
    ...
    observation, self._device, self._task, self._robot_type

So a background thread watching a text file can swap the instruction between one
inference and the next, with no reload and no pause. Whatever writes that file --
a keyboard, Whisper, an LLM -- is then just a front end.

Swapping the attribute alone is not enough to watch, though. Three things made it
impossible to tell which instruction the arm was on:

  * the policy keeps a queue of already-planned actions and only consults the
    instruction when that queue runs dry, so the first second or so after a
    button press is still the previous instruction being carried out;
  * a new instruction begins from wherever the old one left the arm, which is
    mid-reach over some other cube -- a state no training episode ever started
    from, so the new instruction starts out of distribution;
  * nothing reported which instruction was actually being executed, as opposed to
    which one had been written to the file.

All three are handled here. A new instruction ends the current one immediately,
the arm returns to the pose every training episode started from, and only then
does the new instruction begin. Each instruction gets a visible beginning and a
visible end, and what the arm is doing is written to a status file the sender can
display.

Enable with LIVE_TASK_FILE pointing at the file to watch, and LIVE_HOME_DATASET
at the dataset whose start pose to return to between instructions:

    $env:LIVE_TASK_FILE = "E:\Repo\so101-build\task.txt"
    $env:LIVE_HOME_DATASET = "datasets\so101_mix1"
"""

import json
import os
import threading
import time
from pathlib import Path

_POLL_S = 0.25

# STOP returns the arm home and then idles, which is the right end to a task but
# the wrong response to "it is about to hit something". FREEZE is the other one:
# no homing, no motion, hold position from this tick on. Sent as a sentinel
# rather than an empty file so it stays distinguishable from STOP.
_FREEZE = "!freeze"

# Returning home is a position move, not a policy rollout.
#
# The speed is per SECOND, not per tick, and the first version got this wrong:
# 2.0 units per tick with a 5 second deadline is 300 units of travel at 30 Hz but
# only 84 at 8.4 Hz, which is what this control loop actually runs at once two
# cameras and a policy are in it. Half of all recorded poses are more than 70
# units from home and mid-reach is over 130, so homing kept timing out partway
# and handing a half-homed, out-of-distribution pose to the policy. Anything
# measured per tick is measured in a unit whose length nobody controls.
_HOME_SPEED = 45.0        # units per second asked for; the servo decides the rest
_HOME_MAX_STEP = 14.0     # per-tick ceiling, under the robot's max_relative_target
# Arrival is judged per joint, against the band that joint actually occupied at
# the start of the recorded episodes -- not against one global number. A single
# tolerance is wrong in both directions at once: shoulder_lift started within 0.6
# units across 150 episodes while wrist_roll spanned 50. A global 3.5 declared
# elbow_flex "not home" at 4.2 units off its median, a pose sitting comfortably
# inside the 75.4-to-96.8 band the policy was trained from, and the arm was
# therefore never allowed to start.
_HOME_BAND_PCT = (2, 98)  # the start-pose range counted as in distribution
_HOME_BAND_PAD = 2.0      # widened for what the servo can actually hold

# Give up on no PROGRESS, not on a time budget. A budget needs a speed to divide
# by, and every guess at that speed has been wrong: the loop was assumed to run at
# 30 Hz and ran at 8.4, then at 3.6; and the arm was assumed to move at the 45
# units per second it is asked for, while the servo actually delivered 8.7. A
# 79-unit move was then abandoned after 4.8 seconds when it needed nine, and the
# arm was left halfway home. Whether a joint is still closing on its target needs
# no estimate of anything.
_HOME_STALL_S = 4.0       # no measurable progress for this long means stuck
_HOME_PROGRESS = 1.0      # units of improvement that count as progress
_HOME_CEILING_S = 90.0    # backstop, so nothing can wait forever

# Commanding a joint to its target is not the same as it arriving there. Under
# gravity the servo settles short and stays short: elbow_flex stopped 5.5 units
# out and no amount of repeating the same command moved it, because the command
# WAS the target. So when a joint stops improving while still out of tolerance,
# push the command past the target by a growing bias -- ordinary integral action.
# It is bounded, and a joint that is still not moving with the bias fully wound on
# is genuinely blocked, which is what stuck should mean.
_HOME_BIAS_STEP = 3.0     # added per stall, in the direction of the error
_HOME_BIAS_MAX = 12.0     # never command more than this past the target

# The path home, in stages. Moving every joint at once draws a straight line in
# joint space from wherever the policy left the arm, and that line runs through
# the table and through the wrist camera -- a camera has already been hit and
# screws shaken loose that way. So: fold the arm up first, then turn the wrist,
# and only then swing the base. Each stage moves the joints it names and holds
# every other joint where it is measured.
_STAGES = (
    ("lift",  ("shoulder_lift.pos", "elbow_flex.pos")),
    ("wrist", ("wrist_flex.pos", "wrist_roll.pos")),
    ("home",  None),                       # None: all joints, to the home pose
)

# Optional intermediate pose for the first two stages, captured from the arm
# itself with `robot.ps1 waypoint`. Without it those stages aim at the home
# pose's own values for those joints, which is already the folded-up rest
# posture. Capturing one lets the arm be lifted higher, or the wrist parked at a
# specific angle, without anyone guessing a number.
_WAYPOINT_FILE = Path(__file__).resolve().parent / "home_waypoint.json"

_engines: list = []
# "gen" counts writes to the file, not distinct instructions. Pressing the same
# button twice is a deliberate act -- restart this instruction -- and comparing
# only the text made the second press do nothing at all, silently. That is worst
# exactly when it matters: after homing has given up, the panel says to try
# again, and trying again is the one thing that has no effect.
_state = {"task": None, "mtime": None, "path": None, "gen": 0}
_status = {"path": None, "last": None}
# "active" starts as "" rather than None so startup is not itself a transition:
# with None the first tick sees the instruction change from None to "" and drives
# a homing move before anything has been asked for, which is the arm moving on
# its own again in a new disguise.
_ctl = {"active": "", "mode": "idle", "t0": 0.0, "tprev": 0.0, "best": 0.0,
        "best_at": 0.0, "bias": {}, "band": {}, "stuck": False, "home": None,
        "waypoint": None, "stage": 0, "gen": 0, "checked": False}


def _read(path: Path) -> str | None:
    try:
        # utf-8-sig also accepts a plain UTF-8 file; it just removes a BOM if one
        # is there. Anything may write this file -- PowerShell 5.1 adds a BOM by
        # default -- and a leading BOM would silently change the instruction.
        text = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


def _watch(path: Path) -> None:
    """Poll the file and push any change onto every live inference engine."""
    while True:
        try:
            mtime = path.stat().st_mtime if path.exists() else None
            if mtime != _state["mtime"]:
                _state["mtime"] = mtime
                task = _read(path)
                # An empty file clears the instruction rather than being ignored:
                # that is what makes a STOP button possible, and with no
                # instruction the engines produce no action at all.
                _state["task"] = task
                _state["gen"] += 1
                for eng in _engines:
                    # sync and rtc engines name it differently
                    for attr in ("_task", "task"):
                        if hasattr(eng, attr):
                            setattr(eng, attr, task or "")
                print(f"[live-task] -> {task!r}" if task else "[live-task] -> idle", flush=True)
        except Exception as e:  # never let the watcher kill the run
            print(f"[live-task] watcher error: {type(e).__name__}: {e}", flush=True)
        time.sleep(_POLL_S)


def _log(msg: str) -> None:
    """One line per state change, on the console, with numbers in it.

    The previous version reported its state only into a file the web page read,
    so from the terminal there was no way to tell a transition that ran from one
    that never fired -- which is the same as not having built it.
    """
    print(f"[live-task] {msg}", flush=True)


def _short(task: str) -> str:
    """'red out' / 'yellow -> bowl' / 'STOP', for a log line that stays readable."""
    if task == _FREEZE:
        return "FREEZE"
    if not task:
        return "STOP"
    words = task.lower().split()
    colour = next((w for w in words if w in
                   ("red", "blue", "yellow", "green", "purple", "pink")), "?")
    return f"{colour} -> bowl" if " in the black bowl" in task.lower() else f"{colour} out"


def _worst(engine, obs_frame) -> float:
    """How far the arm is from home, in the joint that is furthest off."""
    try:
        names = engine._dataset_features["observation.state"]["names"]
        cur = dict(zip(names, [float(v) for v in obs_frame["observation.state"]]))
        return max(abs(_ctl["home"][k] - cur[k]) for k in engine._ordered_action_keys)
    except Exception:
        return float("nan")


def _report(mode: str, task: str) -> None:
    """Publish what the arm is actually doing, for whoever sent the instruction.

    The sender knows what it wrote; it has no way to know when -- or whether --
    that instruction reached the arm. Without this the only feedback is watching
    the arm and guessing, which is what makes a wrong grasp impossible to
    attribute to either the policy or the button.
    """
    if _status["path"] is None:
        return
    line = json.dumps({"mode": mode, "task": task})
    if line == _status["last"]:
        return
    _status["last"] = line
    try:
        _status["path"].write_text(line, encoding="utf-8")
    except OSError:
        pass


def _load_home(engine) -> dict | None:
    """Median start pose across the training episodes, keyed by joint name.

    The same target home.py parks at before the run: the policy has only ever
    seen episodes that began there, so it is the one pose from which a fresh
    instruction is in distribution.
    """
    root = os.environ.get("LIVE_HOME_DATASET")
    if not root:
        print("[live-task] LIVE_HOME_DATASET not set -- instructions switch "
              "immediately, without returning home first", flush=True)
        return None
    try:
        import home as home_mod

        target, starts = home_mod.target_pose(str(Path(root)))
        names = engine._dataset_features["observation.state"]["names"]
        if len(names) != len(target):
            raise ValueError(f"{len(names)} state names vs {len(target)} dataset columns")
        pose = {n: float(v) for n, v in zip(names, target)}
        missing = [k for k in engine._ordered_action_keys if k not in pose]
        if missing:
            raise ValueError(f"no home value for action keys {missing}")
        _log(f"return-to-home pose from {root}: "
             + "  ".join(f"{k.split('.')[0]} {v:.0f}" for k, v in pose.items()))
        import numpy as np

        lo, hi = np.percentile(starts, _HOME_BAND_PCT, axis=0)
        _ctl["band"] = {n: (float(min(l, t)) - _HOME_BAND_PAD,
                            float(max(h, t)) + _HOME_BAND_PAD)
                        for n, l, h, t in zip(names, lo, hi, target)}
        _log("in-distribution band per joint: "
             + "  ".join(f"{k.split('.')[0]} {v[0]:.0f}..{v[1]:.0f}"
                         for k, v in _ctl["band"].items()))
        _ctl["waypoint"] = _load_waypoint(pose)
        return pose
    except Exception as e:
        print(f"[live-task] cannot load home pose ({type(e).__name__}: {e}) -- "
              "instructions will switch without returning home", flush=True)
        return None


def _load_waypoint(home: dict) -> dict | None:
    """The captured intermediate pose, if there is one.

    Captured from the arm rather than typed in, because the number that matters
    is "the angle where the wrist clears everything on this desk", which is a
    fact about the desk. Nobody can read it off a dataset.
    """
    if not _WAYPOINT_FILE.exists():
        return None
    try:
        wp = json.loads(_WAYPOINT_FILE.read_text(encoding="utf-8"))
        wp = {k: float(v) for k, v in wp.items() if k in home}
        if not wp:
            raise ValueError("no joint names in common with the home pose")
        _log(f"waypoint from {_WAYPOINT_FILE.name}: "
             + "  ".join(f"{k.split('.')[0]} {v:.0f}" for k, v in wp.items()))
        return wp
    except Exception as e:
        _log(f"ignoring {_WAYPOINT_FILE.name} ({type(e).__name__}: {e}); "
             "lifting to the home pose's own values instead")
        return None


def _off(joint: str, value: float) -> float:
    """How far outside its in-distribution band this joint is. Zero means home.

    Arrival is a question about the training data, not about a number someone
    chose: the policy started episodes anywhere inside this band, so anywhere
    inside it is a legitimate place to start one.
    """
    lo, hi = _ctl["band"].get(joint, (float("-inf"), float("inf")))
    return max(0.0, lo - value, value - hi)


def _stage_target(engine, cur: dict) -> tuple[str, dict, tuple]:
    """Where this stage is going, and which joints have to arrive for it to end.

    Driving every joint at once traces a straight line in joint space from
    wherever the arm was left, and that line goes through the table and through
    the wrist camera. The arm has to be folded up before anything swings.
    """
    label, movers = _STAGES[_ctl["stage"]]
    home, wp = _ctl["home"], _ctl["waypoint"] or _ctl["home"]
    keys = engine._ordered_action_keys
    if movers is None:                                  # last stage: everything
        return label, {k: home[k] for k in keys}, tuple(keys)
    # Joints outside this stage hold their measured position, so the stage moves
    # only what it names.
    movers = tuple(k for k in movers if k in keys)
    return label, {k: (wp.get(k, home[k]) if k in movers else cur[k])
                   for k in keys}, movers


def _home_action(engine, obs_frame):
    """One step of the staged return-to-home move, or None when it is over.

    Sets _ctl["stuck"] if it gave up rather than arrived, because those two need
    opposite responses: arriving means start the instruction, giving up means do
    not, since the pose it is stuck in is the out-of-distribution one.
    """
    import torch

    names = engine._dataset_features["observation.state"]["names"]
    cur = dict(zip(names, [float(v) for v in obs_frame["observation.state"]]))
    keys = engine._ordered_action_keys
    now = time.perf_counter()

    while True:
        label, target, movers = _stage_target(engine, cur)
        # Distance from the band, not from the median: a joint sitting anywhere
        # the recorded episodes started from has arrived.
        worst = max(_off(k, cur[k]) for k in movers)
        if worst > 0:
            break
        # Stage done. Each stage gets its own patience, so a long leg is not
        # judged by how long an earlier short one took.
        _ctl["stage"] += 1
        if _ctl["stage"] >= len(_STAGES):
            return None
        nlabel, ntarget, nmovers = _stage_target(engine, cur)
        far = max(abs(ntarget[k] - cur[k]) for k in nmovers)
        _ctl["best"], _ctl["best_at"], _ctl["bias"] = float("inf"), now, {}
        _log(f"  homing stage {_ctl['stage'] + 1}/{len(_STAGES)} {nlabel}: "
             f"{far:.0f} units")

    # Progress, not a clock. Any real improvement resets the patience; only a
    # joint that has stopped closing on its target counts as stuck.
    if worst < _ctl["best"] - _HOME_PROGRESS:
        _ctl["best"], _ctl["best_at"] = worst, now
    stalled = now - _ctl["best_at"] > _HOME_STALL_S
    if stalled:
        # Wind the bias on before calling it stuck: the joint may simply be
        # settling short of a command that is exactly its target.
        room = [k for k in movers
                if _off(k, cur[k]) > 0
                and abs(_ctl["bias"].get(k, 0.0)) < _HOME_BIAS_MAX]
        if room:
            for k in room:
                step = _HOME_BIAS_STEP if target[k] > cur[k] else -_HOME_BIAS_STEP
                _ctl["bias"][k] = _ctl["bias"].get(k, 0.0) + step
            _ctl["best_at"] = now
            _log(f"  {label}: settling short, pushing "
                 + ", ".join(f"{k.split('.')[0]} by {_ctl['bias'][k]:+.0f}"
                             for k in room))
            stalled = False
    if stalled or now - _ctl["t0"] > _HOME_CEILING_S:
        joint = max(movers, key=lambda k: abs(target[k] - cur[k]))
        why = ("will not move even pushed past the target" if stalled
               else "still going after 90s")
        _log(f"gave up homing in stage {label} after {now - _ctl['t0']:.1f}s "
             f"({why}): {joint} still {worst:.1f} units off. Holding still "
             "rather than starting the instruction from a pose the policy "
             "never saw.")
        _ctl["stuck"] = True
        return None

    # Distance per second, converted here into distance for THIS tick using the
    # time the last tick actually took. The move then covers the same ground in
    # the same wall time whatever rate the control loop happens to run at.
    dt = min(now - _ctl["tprev"], 0.25)
    _ctl["tprev"] = now
    limit = min(_HOME_SPEED * dt, _HOME_MAX_STEP)
    if _ctl["bias"]:
        # Once a joint has stalled, the step itself is the problem, not just where
        # it points: a command smaller than the joint's stiction simply does not
        # move it, and the speed-derived step shrinks as the error does. Use the
        # full allowance to break it loose.
        limit = _HOME_MAX_STEP
    out = []
    for k in keys:
        e = target[k] - cur[k] + _ctl["bias"].get(k, 0.0)
        out.append(cur[k] + max(-limit, min(limit, e)))
    return torch.tensor(out)


_times: list = []


def _timing(dt: float) -> None:
    """Report where the tick actually goes, every hundred policy calls.

    Policy time alone is not the number that matters. The chunk is 50 actions
    long, so the rate at which ticks actually happen decides how long the arm
    runs blind between one look at the cameras and the next: 1.7 s at the 30 Hz
    the data was recorded at, but 6 s at 8 Hz. lerobot's own "running slower"
    warning fires on single ticks and is rate limited, so it cannot answer this.
    Measured here instead of inferred.
    """
    now = time.perf_counter()
    if not _times:
        _timing.started = now                              # type: ignore[attr-defined]
    _times.append(dt)
    if len(_times) < 100:
        return
    xs = sorted(_times)
    wall = now - getattr(_timing, "started", now)
    _times.clear()
    med, worst = xs[len(xs) // 2], xs[-1]
    hz = len(xs) / wall if wall > 0 else 0.0
    _log(f"policy call over 100 ticks: median {med * 1000:.0f} ms, "
         f"worst {worst * 1000:.0f} ms, total {sum(xs):.1f}s of "
         f"{len(xs)} ticks")
    _log(f"loop ran at {hz:.1f} Hz ({wall:.1f}s for 100 ticks); "
         f"a 50-action chunk takes {50 / hz:.1f}s at that rate, "
         f"against 1.7s in the training data")


def _install_engine_patch() -> None:
    """Give each instruction a clean beginning and a clean end.

    Left alone the engine returns whatever its planned chunk says and only looks
    at the instruction once that chunk is exhausted, so instructions blur into one
    another and a blank instruction still produced motion. This wraps get_action
    in a three-state machine -- idle, homing, running -- and reports which state
    it is in.
    """
    from lerobot.rollout.inference import rtc, sync

    for mod, cls_name in ((sync, "SyncInferenceEngine"), (rtc, "RTCInferenceEngine")):
        cls = getattr(mod, cls_name, None)
        if cls is None or getattr(cls, "_idle_guard_patched", False):
            continue
        orig_get = cls.get_action

        def get_action(self, obs_frame, *a, _orig=orig_get, **kw):
            if obs_frame is None:
                return None
            want = getattr(self, "_task", None) or getattr(self, "task", None) or ""
            want = str(want).strip()

            if not _ctl["checked"]:
                _ctl["checked"] = True
                _log(f"patched {type(self).__name__}.get_action")
                _ctl["home"] = _load_home(self)

            if want != _ctl["active"] or _state["gen"] != _ctl["gen"]:
                _ctl["gen"] = _state["gen"]
                # End the old instruction here, not when its chunk happens to run
                # out. reset() drops the queued actions along with the processor
                # state, so the next inference is planned from the pose the arm is
                # actually in rather than the one the old chunk assumed.
                _ctl["active"] = want
                _ctl["mode"] = "homing" if _ctl["home"] is not None else "running"
                _ctl["t0"] = _ctl["tprev"] = time.perf_counter()
                _ctl["stuck"] = False
                _ctl["stage"] = 0
                try:
                    self.reset()
                except Exception:
                    pass
                if want == _FREEZE:
                    _ctl["mode"] = "frozen"
                    _log("FREEZE: holding position from this tick, no homing")
                elif _ctl["mode"] == "homing":
                    far = _worst(self, obs_frame)
                    _ctl["best"], _ctl["best_at"] = float("inf"), _ctl["t0"]
                    _ctl["bias"] = {}
                    _log(f"{_short(want)}: broke off, {far:.0f} units from home, "
                         f"going back in {len(_STAGES)} stages")
                else:
                    _log(f"{_short(want)}: starting at once, no home pose loaded")

            if _ctl["mode"] == "frozen":
                _report("frozen", "")
                return None

            if _ctl["mode"] == "homing":
                try:
                    act = _home_action(self, obs_frame)
                except Exception as e:
                    # Better to switch instructions abruptly than to abort the
                    # run: a policy trained without proprioception has no state
                    # in its frames, and there is nothing to interpolate from.
                    print(f"[live-task] homing failed ({type(e).__name__}: {e}); "
                          "switching instructions without it", flush=True)
                    _ctl["home"] = None
                    act = None
                if act is not None:
                    _report("homing", want)
                    return act
                if _ctl["stuck"]:
                    _ctl["mode"] = "stuck"
                else:
                    _log(f"home reached in {time.perf_counter() - _ctl['t0']:.1f}s "
                         f"(worst {_worst(self, obs_frame):.1f}); now "
                         f"{'running ' + _short(want) if want else 'idle'}")
                    _ctl["mode"] = "running"
                    try:
                        self.reset()
                    except Exception:
                        pass

            if _ctl["mode"] == "stuck":
                # Homing did not finish, so the arm is somewhere no episode ever
                # started from. Running the policy from here is what produces the
                # thrashing that hits the wrist camera. Wait for a new
                # instruction instead; STOP retries the move.
                _report("stuck", want)
                return None

            if not want:
                # No instruction means no action at all. Seeding the file with a
                # default meant the arm began working the moment it started, on
                # whatever task happened to be the default -- which was the old
                # one. Returning None is already the "nothing this tick" path
                # through send_next_action, so the arm holds position.
                _report("idle", "")
                return None
            _report("running", want)
            # Time the policy call itself. The control loop ran at 2 Hz while the
            # policy drove and kept up while it idled, which narrows the cost to
            # this call or to what surrounds it -- and those need different fixes.
            # Guessing between them has already cost a night, so measure.
            t0 = time.perf_counter()
            out = _orig(self, obs_frame, *a, **kw)
            _timing(time.perf_counter() - t0)
            return out

        cls.get_action = get_action
        cls._idle_guard_patched = True


def apply(path: str | None = None, verbose: bool = True) -> None:
    """Patch the inference engines so their instruction follows `path`."""
    path = path or os.environ.get("LIVE_TASK_FILE")
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    _install_engine_patch()

    from lerobot.rollout.inference import rtc, sync

    for mod, cls_name in ((sync, "SyncInferenceEngine"), (rtc, "RTCInferenceEngine")):
        cls = getattr(mod, cls_name, None)
        if cls is None or getattr(cls, "_live_task_patched", False):
            continue
        orig_init = cls.__init__

        def __init__(self, *a, _orig=orig_init, **kw):
            _orig(self, *a, **kw)
            _engines.append(self)
            # The file wins over --task, including when the file is empty.
            # lerobot-rollout takes one --task string for the whole run and has
            # no way to say "nothing", so robot.ps1 passes its default even in
            # live mode. Blanking task.txt was not enough: the engine was
            # constructed with that default and started working on it with
            # nothing pressed, which is the old put-it-in-the-bowl task acting on
            # its own all over again.
            was = getattr(self, "_task", None) or getattr(self, "task", None)
            for attr in ("_task", "task"):
                if hasattr(self, attr):
                    setattr(self, attr, _state["task"] or "")
            if was and not _state["task"]:
                _log(f"ignoring --task {was!r}; idle until you send an instruction")

        cls.__init__ = __init__
        cls._live_task_patched = True

    # Seed from the file so a task written before launch is picked up.
    _state["path"] = p
    _state["task"] = _read(p)
    _state["mtime"] = p.stat().st_mtime if p.exists() else None
    _status["path"] = p.with_suffix(".status")
    _report("idle", "")

    threading.Thread(target=_watch, args=(p,), daemon=True).start()
    if verbose:
        print(f"[live-task] watching {p}  (current: {_state['task']!r})", flush=True)
