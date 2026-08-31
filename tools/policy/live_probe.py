"""Feed the policy ONE live observation and inspect the chunk it plans.

WHY
    Every offline check so far fed observations taken from the dataset, and
    they all passed: step-1 MAE ~1 unit, chunk error at step 100 ~2 units,
    camera mapping correct, appearance in distribution, start pose in
    distribution, max_relative_target never binding.

    The single untested link is the live path: camera -> policy, with no mp4
    encode/decode in between. Training only ever saw H.264-decoded frames.

    So take one real observation from the robot right now, plan a chunk, and
    look at what it actually wants to do. Then do the same from a dataset
    observation and compare the two trajectories side by side.

    "Oscillates up and down without approaching the object" should be visible
    here as a chunk that swings a joint back and forth instead of moving
    monotonically toward a target.

ASCII output only.

Usage:
    python live_probe.py --port COM8
"""

import argparse
import os
import sys

# This lives under tools/ but imports modules kept at the repo root (Python puts
# the script's own directory on the path, not the working directory).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json

import numpy as np
import torch


def describe(chunk, names):
    """Summarise a planned trajectory: net motion, total path, reversals."""
    print("%-15s %9s %9s %9s %9s" % ("joint", "start", "end", "net", "path"))
    for j, nm in enumerate(names):
        c = chunk[:, j]
        d = np.diff(c)
        net = c[-1] - c[0]
        path = np.abs(d).sum()
        print("%-15s %9.2f %9.2f %9.2f %9.2f" % (nm, c[0], c[-1], net, path))
    print()
    # path / |net| >> 1 means the plan wanders instead of going somewhere
    for j, nm in enumerate(names):
        c = chunk[:, j]
        net = abs(c[-1] - c[0])
        path = np.abs(np.diff(c)).sum()
        ratio = path / net if net > 0.5 else float("inf") if path > 2 else 0.0
        flag = "  <-- wanders" if ratio > 3 and path > 3 else ""
        print("%-15s path/|net| = %6.1f%s" % (nm, ratio, flag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--ckpt", default="policies/act_v3_100k")
    a = ap.parse_args()

    import lerobot_patch
    lerobot_patch.apply(verbose=False)

    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    m = json.load(open("camera_map.json"))
    cams = {
        "top": OpenCVCameraConfig(index_or_path=int(m["top"]), width=640, height=480, fps=30),
        "wrist": OpenCVCameraConfig(index_or_path=int(m["wrist"]), width=640, height=480, fps=30),
    }
    dev = SO101Follower(SO101FollowerConfig(
        port=a.port, id="my_follower", max_relative_target=None, cameras=cams))
    dev.connect(calibrate=False)

    policy = ACTPolicy.from_pretrained(a.ckpt)
    policy.to("cpu"); policy.eval()
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=a.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})

    names = list(dev.bus.motors.keys())
    try:
        obs = dev.get_observation()
    finally:
        dev.disconnect()

    state = torch.tensor([[obs[f"{n}.pos"] for n in names]], dtype=torch.float32)
    def img(k):
        arr = obs[k]                      # HWC uint8 RGB
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
        return t[None]

    batch = {
        "observation.state": state,
        "observation.images.top": img("top"),
        "observation.images.wrist": img("wrist"),
    }

    with torch.no_grad():
        policy.reset()
        o = pre(batch)
        live_chunk = np.array([post(policy.select_action(o))[0].numpy() for _ in range(100)])

    print("=== LIVE observation ===")
    print("state:", np.round(state.numpy()[0], 2))
    print()
    describe(live_chunk, names)

    # same treatment on a dataset observation, as a control
    ds = LeRobotDataset("local/so101_pickplace", root="datasets/so101_v3")
    s = ds[0]
    dbatch = {
        "observation.state": s["observation.state"][None],
        "observation.images.top": s["observation.images.top"][None],
        "observation.images.wrist": s["observation.images.wrist"][None],
    }
    with torch.no_grad():
        policy.reset()
        o = pre(dbatch)
        ds_chunk = np.array([post(policy.select_action(o))[0].numpy() for _ in range(100)])

    print()
    print("=== DATASET observation (control, episode 0 frame 0) ===")
    print("state:", np.round(s["observation.state"].numpy(), 2))
    print()
    describe(ds_chunk, names)


if __name__ == "__main__":
    main()
