r"""does_it_start.py on the scene in front of the cameras right now.

does_it_start.py feeds each checkpoint the opening frames of the recordings.
That answers "can this model leave the rest pose on the data it was trained
on", not "will it leave on the desk as it is tonight". When a model that passed
offline sits still on the robot, the two differ, and the difference is either
the model or the scene. Feeding several checkpoints the same live frames tells
which: if the one that worked before plans to move and the new one does not,
it is the model; if none of them move, it is the scene (light, layout, camera).

The arm must be resting at the start pose and nothing else may hold the
cameras (stop any eval first). Nothing is sent to the arm.

    .\.venv-win\Scripts\python.exe tools\policy\does_it_start_live.py ckpt [ckpt ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = 8
OUT = ROOT / "figures" / "live_start"


def grab(index):
    import cv2                                        # after torch, as with av

    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(30):                               # let exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError(f"camera {index} gave no frame -- is an eval still running?")
    return frame


def travel(chunk):
    """How far the plan takes the arm from where it starts, for the joint that
    goes furthest. Gripper excluded.

    This used to be path length -- the sum of every step's change. That counts
    jitter: a plan that trembles in place by a unit or two per step adds up to
    20-30 over 50 steps, and on 2026-10-16 a model that sat still on the robot
    scored 29 and "moves 100%". Excursion from the start does not accumulate
    trembling: it is the distance the arm would actually travel."""
    arm = chunk[:, :5]
    return float(np.abs(arm - arm[0]).max())


cams = json.loads((ROOT / "camera_map.json").read_text())
import cv2                                            # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
frames = {}
for name in ("top", "wrist"):
    bgr = grab(cams[name])
    cv2.imwrite(str(OUT / f"{name}.jpg"), bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    frames[name] = torch.from_numpy(rgb).permute(2, 0, 1)
print(f"frames saved to {OUT}")

# The arm is homed to the dataset's start pose before every eval, so use that
# pose as the state rather than opening the serial port -- the same target
# robot.ps1 drives to (home.target_pose), not one episode's first frame, which
# on 2026-10-16 differed from it by 40 units of wrist roll.
sys.path.insert(0, str(ROOT))
from home import target_pose                          # noqa: E402

state = torch.tensor(target_pose(str(ROOT / "datasets" / "so101_emptybowl"))[0], dtype=torch.float32)

from lerobot.configs.policies import PreTrainedConfig                          # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

print(f"\n  {'':<34}{'plan travel':>12}{'moves (>5)':>12}")
for name in sys.argv[1:]:
    d = ROOT / "policies" / name
    cfg = PreTrainedConfig.from_pretrained(str(d))
    policy = get_policy_class(cfg.type).from_pretrained(str(d)).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(d),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    want = sorted(k for k in cfg.input_features if ".images." in k)
    obs = {"observation.state": state, "robot_type": "so101_follower",
           "task": "Take all the cubes out of the bowl and put them on the table."}
    for key, src in zip(want, ["top", "wrist"]):
        obs[key] = frames[src]
    nobs = int(getattr(cfg, "n_obs_steps", 1) or 1)
    t = []
    with torch.inference_mode():
        for _ in range(SAMPLES):
            policy.reset()
            if nobs > 1:
                for _ in range(nobs - 1):
                    policy.select_action(pre(dict(obs)))
                first = policy.select_action(pre(dict(obs)))
                chunk = post(torch.stack([first] + list(policy._queues["action"]), dim=1))
            else:
                chunk = post(policy.predict_action_chunk(pre(dict(obs))))
            t.append(travel(chunk.squeeze(0).float().cpu().numpy()))
    t = np.array(t)
    print(f"  {name:<34}{np.median(t):12.1f}{np.mean(t > 5):12.0%}")
    del policy
    torch.cuda.empty_cache()
