r"""Check live_task.py without the robot: real engine class, real loop rate.

Everything here is decided on the arm, in real time, where a wrong answer costs
an evening. This reproduces the decisions in a second, with no servos and no
policy, so the question "did the homing actually run" has an answer that does not
depend on watching.

Two things an earlier version of this test did not do, and both hid a real bug.
It injected a stub class into fake modules, which proved the state machine's
logic and nothing about whether the patch binds to the class the rollout
constructs. And it ticked as fast as the CPU allowed, which hid a homing move
whose speed was written per tick while its deadline was written in seconds: those
agree only at 30 Hz, and the real loop runs at 8.4 Hz with two cameras and a
policy in it.

    .\.venv-win\Scripts\python.exe tools\policy	est_live_task.py
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

# NOT the live task.txt. A run of this test once wrote instructions into the file
# a real rollout was reading, and read that session's arm state back as its own --
# the test could have driven the arm, and its results were meaningless.
TASK = ROOT / "logs" / "test_task.txt"
TASK.parent.mkdir(parents=True, exist_ok=True)
TASK.write_text("", encoding="utf-8")
os.environ["LIVE_TASK_FILE"] = str(TASK)
os.environ["LIVE_HOME_DATASET"] = "datasets\\so101_v3"

import numpy as np

import live_task
live_task.apply()

from lerobot.rollout.inference.sync import SyncInferenceEngine

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
FEATURES = {"observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
            "action": {"dtype": "float32", "shape": (6,), "names": JOINTS}}

HZ = 8.4                      # the rate the real run reported
RED = "Pick the red cube out of the bowl and put it on the table."


class FakePolicy:
    class config:
        use_amp = False

    def reset(self):
        pass

    def select_action(self, obs):
        raise RuntimeError("POLICY RAN")


engine = SyncInferenceEngine(
    policy=FakePolicy(), preprocessor=lambda x: x, postprocessor=lambda x: x,
    dataset_features=FEATURES, ordered_action_keys=list(JOINTS),
    task="Pick the yellow block and put it in the black bowl.",   # robot.ps1 default
    device="cpu", robot_type="so101_follower",
)

print("\n== the file wins over --task ==")
print("  engine._task:", repr(engine._task))
assert engine._task == "", "the --task default survived; the arm would start on it"

import home as home_mod
target = np.array(home_mod.target_pose("datasets/so101_v3")[0], dtype=float)
pose = target.copy()
stuck_joint = None


def tick(slow=1.0, droop=0.0):
    """One control tick at the real loop rate.

    `slow` is the fraction of the commanded step the servo actually delivers.
    The real one delivered about a fifth of what it was asked for, and the give-up
    rule called that a failure because the rule divided a distance by a guessed
    speed. Slow and stuck are different things and have to be tested as such.
    """
    global pose
    time.sleep(1.0 / HZ)
    try:
        out = engine.get_action({"observation.state": pose.copy()})
    except RuntimeError as e:
        return str(e)
    if out is not None:
        want = np.array([float(v) for v in out])
        new = pose + (want - pose) * slow
        if droop:
            # A deadband: commanded moves smaller than this do not happen at all.
            # That is what elbow_flex did on the bench. Large steps tracked fine,
            # and then the last 5.5 units never closed, because as the error
            # shrinks so does the commanded step, and below the deadband the joint
            # simply stops. Repeating the same command cannot help; only asking
            # for a bigger move can.
            if abs(want[2] - pose[2]) < droop:
                new[2] = pose[2]
            else:
                new[2] = want[2]
        if stuck_joint is not None:
            new[stuck_joint] = pose[stuck_joint]      # a servo that will not move
        pose = new
    return out


def home_ok():
    """Is every joint inside the band the recorded episodes started from?"""
    return all(live_task._off(j, pose[i]) == 0 for i, j in enumerate(JOINTS))


def off_report():
    return {j.split(".")[0]: round(live_task._off(j, pose[i]), 1)
            for i, j in enumerate(JOINTS) if live_task._off(j, pose[i]) > 0}


def st():
    return json.loads(TASK.with_suffix(".status").read_text(encoding="utf-8"))


def run_until(done, limit_s=25):
    t0 = time.perf_counter()
    while True:
        out = done(tick())
        if out is not None:
            return out, time.perf_counter() - t0
        assert time.perf_counter() - t0 < limit_s, f"stuck in {st()}"


print("\n== nothing pressed ==")
assert tick() is None and st()["mode"] == "idle", st()
print("  idle, no action sent")

print(f"\n== press red from 130 units out, loop at {HZ} Hz ==")
pose = target.copy(); pose[0] += 130.0
TASK.write_text(RED, encoding="utf-8"); time.sleep(0.6)
_, secs = run_until(lambda o: True if isinstance(o, str) else None)
print(f"  homed in {secs:.1f}s, residual {np.abs(pose - target).max():.1f}, then the policy ran")
assert st()["mode"] == "running", st()
assert home_ok(), f"not in the recorded band: {off_report()}"

print(f"\n== STOP from 185 units out, the worst case in the data ==")
pose = target.copy(); pose[0] += 185.0
TASK.write_text("", encoding="utf-8"); time.sleep(0.6)
_, secs = run_until(lambda o: True if o is None else None)
print(f"  homed in {secs:.1f}s, residual {np.abs(pose - target).max():.1f}, then stopped")
assert st() == {"mode": "idle", "task": ""}, st()
assert home_ok(), f"not in the recorded band: {off_report()}"

print("\n== a SLOW joint must not be mistaken for a stuck one ==")
# The exact failure this guards against, measured on the bench: the loop ran at
# 3.6 Hz, not the 30 assumed, and the servo delivered 8.7 units per second, not
# the 45 it was asked for. A 79-unit move was abandoned after 4.8 seconds when it
# needed nine, and the arm was left halfway home with the panel waiting forever.
pose = target.copy(); pose[1] += 80.0
TASK.write_text(RED, encoding="utf-8"); time.sleep(0.6)
n = 0
while not isinstance(tick(slow=0.2), str):        # a fifth of what was commanded
    n += 1
    assert st()["mode"] != "stuck", (
        f"a slow joint was called stuck after {n} ticks; "
        f"err={np.abs(pose - target).max():.1f} ctl={ {k: v for k, v in live_task._ctl.items() if k in ('mode','stage','best','best_at','t0')} }")
    assert n < 4000, f"never arrived; {np.abs(pose - target).max():.1f} left"
print(f"  crawled home in {n} ticks at a fifth speed, never called stuck")
assert home_ok(), f"not in the recorded band: {off_report()}"

print("\n== a joint that settles SHORT must still be brought home ==")
# The bench failure: elbow_flex stopped 5.5 units out and homing gave up, so the
# arm never actually got home and the panel reported "could not reach the start
# pose" on every press.
pose = target.copy(); pose[2] -= 25.0
TASK.write_text(RED, encoding="utf-8"); time.sleep(0.6)
n = 0
while not isinstance(tick(droop=5.5), str):
    n += 1
    assert st()["mode"] != "stuck", (
        f"gave up on a joint that was only settling short, after {n} ticks, "
        f"{abs(pose[2] - target[2]):.1f} units out")
    assert n < 1500, "never arrived"
print(f"  pushed past the target until it arrived, residual "
      f"{abs(pose[2] - target[2]):.1f}")
assert home_ok(), f"not in the recorded band: {off_report()}"

print("\n== a joint that cannot move at all: refuse to run the policy ==")
stuck_joint = 2
pose = target.copy(); pose[2] += 60.0
TASK.write_text(RED, encoding="utf-8"); time.sleep(0.6)
_, secs = run_until(lambda o: True if st()["mode"] == "stuck" else None)
print(f"  gave up after {secs:.1f}s, status {st()['mode']}")
for _ in range(5):
    assert tick() is None, "it ran the policy from a pose it never reached"
print("  holds still, policy not started")

print("\n== does it fold up BEFORE it swings the base? ==")
# The collision case: arm extended low over the table, base turned right round,
# wrist rolled over. Driving every joint at once sweeps the wrist camera across
# everything in between.
stuck_joint = None
pose = target.copy()
pose[0] += 120.0          # shoulder_pan, swung out
pose[1] += 70.0           # shoulder_lift, down toward the table
pose[2] -= 55.0           # elbow_flex, extended
pose[4] += 90.0           # wrist_roll, rolled over
TASK.write_text(RED, encoding="utf-8"); time.sleep(0.6)

order = []                # when each joint finished moving
start = pose.copy()
while True:
    out = tick()
    if isinstance(out, str):
        break
    for i, j in enumerate(JOINTS):
        if abs(pose[i] - start[i]) > 3.0 and j not in order:
            order.append(j)
    assert st()["mode"] in ("homing", "running"), st()
print("  joints moved in this order:", " then ".join(j.split(".")[0] for j in order))
assert order.index("shoulder_lift.pos") < order.index("shoulder_pan.pos"), \
    "the base swung before the arm folded up"
assert order.index("wrist_roll.pos") < order.index("shoulder_pan.pos"), \
    "the base swung before the wrist was turned"
assert home_ok(), f"not in the recorded band: {off_report()}"
print(f"  arrived home, residual {np.abs(pose - target).max():.1f}")

print("\n== FREEZE while moving: does it stop this tick, without homing? ==")
stuck_joint = None
pose = target.copy(); pose[0] += 150.0
TASK.write_text(RED, encoding="utf-8"); time.sleep(0.6)
tick(); tick()                                   # homing under way
moving = pose.copy()
assert np.abs(moving - target).max() > 3.5, "the test needs the arm away from home"
TASK.write_text("!freeze", encoding="utf-8"); time.sleep(0.6)
for _ in range(6):
    assert tick() is None, "FREEZE still commanded motion"
assert st()["mode"] == "frozen", st()
assert np.array_equal(pose, moving), "the arm moved after FREEZE"
print(f"  froze {np.abs(pose - target).max():.0f} units from home and stayed there")

print("\n== STOP after FREEZE sends it home ==")
TASK.write_text("", encoding="utf-8"); time.sleep(0.6)
_, secs = run_until(lambda o: True if st()["mode"] == "idle" else None)
print(f"  homed in {secs:.1f}s, residual {np.abs(pose - target).max():.1f}")
assert home_ok(), f"not in the recorded band: {off_report()}"

print("\n== home.py takes the same staged path ==")
# home.py runs before every evaluation, from wherever the previous run left the
# arm. With one instruction per run it is the ONLY thing that moves the arm
# across the desk, so the staging matters more here than in live mode.
JN = [j[:-4] for j in JOINTS]
tgt = np.array(target, dtype=float)
low = tgt + np.array([120.0, 70.0, -55.0, -23.0, 90.0, 29.0])   # low, swung out
seen, prev = [], low.copy()
for label, goal, idx in home_mod.plan_stages(JN, low, tgt):
    moved = [JN[i] for i in range(len(JN)) if abs(goal[i] - prev[i]) > 1e-9]
    print(f"  {label:<6} moves {moved}")
    seen += moved
    prev = goal.copy()
assert seen.index("shoulder_lift") < seen.index("shoulder_pan"), "base swings first"
assert seen.index("wrist_roll") < seen.index("shoulder_pan"), "base swings first"
assert np.allclose(prev, tgt), "the last leg does not reach the start pose"

wp = {"shoulder_lift.pos": -110.0, "wrist_roll.pos": -30.0}
plan = home_mod.plan_stages(JN, low, tgt, wp)
assert plan[0][1][1] == -110.0, "the captured waypoint was ignored on the lift"
assert plan[1][1][4] == -30.0, "the captured waypoint was ignored on the wrist"
assert np.allclose(plan[-1][1], tgt), "the last leg does not reach the start pose"
print("  a captured waypoint overrides the lift and wrist legs, home is unchanged")

print("\nALL OK on the real engine class, at the real loop rate")
