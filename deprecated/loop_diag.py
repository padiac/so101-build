"""Is the policy commanding motion, or is the arm failing to follow?

eval_loop.py issues ~410 commands over 15 s and the arm does not visibly move,
while the same checkpoint driven through lerobot's own rollout does move. Those
two possibilities need different fixes, so print both sides: what was commanded
and where the arm actually is.

Also compares the observation built here against the dataset's, since a
mismatch in image scaling or channel order would leave the policy looking at
something it never saw in training.

ASCII output only.
"""

import argparse
import json
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--ckpt", default="policies/act_v2_100k")
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()

    import lerobot_patch
    lerobot_patch.apply(verbose=False)

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = json.load(open("camera_map.json"))
    be = Cv2Backends.MSMF.value
    cams = {k: OpenCVCameraConfig(index_or_path=int(m[k]), width=640, height=480,
                                  fps=30, backend=be) for k in ("top", "wrist")}

    policy = ACTPolicy.from_pretrained(a.ckpt).to(dev).eval()
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=a.ckpt,
        preprocessor_overrides={"device_processor": {"device": dev}})

    robot = SO101Follower(SO101FollowerConfig(
        port=a.port, id="my_follower", max_relative_target=15.0, cameras=cams))
    robot.connect(calibrate=False)
    names = list(robot.bus.motors.keys())

    try:
        obs = robot.get_observation()

        # --- what does the live observation look like next to the dataset's? ---
        ds = LeRobotDataset("local/v2", root="datasets/so101_v2")
        s0 = ds[0]
        print("=== observation comparison ===")
        for cam in ("top", "wrist"):
            live = np.asarray(obs[cam])
            tr = s0["observation.images.%s" % cam]
            print("  %-6s live  shape %-16s dtype %-8s range %.3f..%.3f"
                  % (cam, live.shape, live.dtype, live.min() / 255.0, live.max() / 255.0))
            print("  %-6s train shape %-16s dtype %-8s range %.3f..%.3f"
                  % ("", tuple(tr.shape), tr.dtype, float(tr.min()), float(tr.max())))
        st = np.array([obs["%s.pos" % n] for n in names])
        print("  state live :", np.round(st, 2))
        print("  state train:", np.round(s0["observation.state"].numpy(), 2))
        print()

        # --- run and log commanded vs measured ---
        print("=== commanded vs measured ===")
        print("%6s %-14s %-40s %-40s" % ("t", "joint", "commanded", "measured"))
        t0 = time.perf_counter()
        policy.reset()
        rows = []
        while time.perf_counter() - t0 < a.seconds:
            obs = robot.get_observation()
            st = np.array([obs["%s.pos" % n] for n in names])
            b = {"observation.state": torch.tensor([st], dtype=torch.float32)}
            for cam in ("top", "wrist"):
                arr = np.ascontiguousarray(obs[cam])
                b["observation.images.%s" % cam] = (
                    torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)[None]
            with torch.no_grad():
                act = post(policy.select_action(pre(b)))[0].cpu().numpy()
            robot.send_action({"%s.pos" % n: float(act[i]) for i, n in enumerate(names)})
            rows.append((time.perf_counter() - t0, act.copy(), st.copy()))
            time.sleep(1 / 30)

        for t, act, st in rows[::30]:
            print("%6.1f %-14s %-40s %-40s"
                  % (t, "", np.array2string(act, precision=1, suppress_small=True),
                     np.array2string(st, precision=1, suppress_small=True)))
        A = np.array([r[1] for r in rows])
        S = np.array([r[2] for r in rows])
        print()
        print("%-14s %12s %12s %12s" % ("joint", "cmd range", "meas range", "mean |cmd-meas|"))
        for i, n in enumerate(names):
            print("%-14s %12.2f %12.2f %12.2f"
                  % (n, A[:, i].max() - A[:, i].min(), S[:, i].max() - S[:, i].min(),
                     np.abs(A[:, i] - S[:, i]).mean()))
        print()
        print("cmd range near zero  -> the policy is asking for nothing")
        print("cmd range large but meas range near zero -> the arm is not following")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
