#!/usr/bin/env python3
"""Run the policy N times in one session, homing between attempts.

Starting robot.ps1 eval once per attempt spends most of the time connecting --
serial handshake, two cameras, loading 52M weights. This connects once and
loops, so the only pause between attempts is the arm moving back to its start
pose, which doubles as the window for repositioning the block.

After each attempt it asks for a score on the staged scale used in
https://huggingface.co/blog/sherryxychen/train-act-on-so-101 :

    0.0  nothing useful      0.2  reached the block
    0.4  grasped it          0.7  carried it over the bowl
    0.8  released            1.0  block in the bowl

Scoring every attempt is the point. "It sort of moved" cannot tell you whether
the next dataset is better or worse; a mean over ten attempts can.

ASCII output only.

Usage:
    python eval_loop.py --port COM8
    python eval_loop.py --port COM8 --trials 10 --seconds 20
"""

import argparse
import json
import time

import numpy as np
import torch

VALID = {"0": 0.0, "0.2": 0.2, "0.4": 0.4, "0.7": 0.7, "0.8": 0.8, "1": 1.0, "1.0": 1.0}
STAGES = "0=nothing  0.2=reached  0.4=grasped  0.7=carried  0.8=released  1=in bowl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--ckpt", default="policies/act_v2_100k")
    ap.add_argument("--dataset", default="datasets/so101_v2")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-rel", type=float, default=15.0)
    ap.add_argument("--no-score", action="store_true", help="skip the score prompt")
    ap.add_argument("--auto", action="store_true",
                    help="run straight through: no Enter, no scoring; the homing between attempts is your window to move the block")
    ap.add_argument("--gap", type=float, default=2.0,
                    help="seconds to wait after homing before the policy takes over")
    a = ap.parse_args()

    import lerobot_patch
    lerobot_patch.apply(verbose=False)

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from home import target_pose

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = json.load(open("camera_map.json"))
    # MSMF must match what cams.py resolve used, or the indices mean different
    # physical cameras (README #29)
    be = Cv2Backends.MSMF.value
    cams = {
        "top": OpenCVCameraConfig(index_or_path=int(m["top"]), width=640, height=480,
                                  fps=30, backend=be),
        "wrist": OpenCVCameraConfig(index_or_path=int(m["wrist"]), width=640, height=480,
                                    fps=30, backend=be),
    }
    print("cameras: top=index %s  wrist=index %s" % (m["top"], m["wrist"]))

    tgt, _ = target_pose(a.dataset)
    print("policy : %s   device %s" % (a.ckpt, dev))

    policy = ACTPolicy.from_pretrained(a.ckpt).to(dev).eval()
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=a.ckpt,
        preprocessor_overrides={"device_processor": {"device": dev}})
    print("chunk_size %s   n_action_steps %s" % (cfg.chunk_size, cfg.n_action_steps))

    robot = SO101Follower(SO101FollowerConfig(
        port=a.port, id="my_follower", max_relative_target=a.max_rel, cameras=cams))
    robot.connect(calibrate=False)
    names = list(robot.bus.motors.keys())
    dt = 1.0 / a.fps

    def obs_batch(obs):
        st = torch.tensor([[obs["%s.pos" % n] for n in names]], dtype=torch.float32)
        out = {"observation.state": st}
        for cam in ("top", "wrist"):
            arr = np.ascontiguousarray(obs[cam])
            out["observation.images.%s" % cam] = (
                torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)[None]
        return out

    def go_home(timeout=25.0, step=1.5, tol=3.5):
        t0 = time.perf_counter()
        while True:
            obs = robot.get_observation()
            cur = np.array([obs["%s.pos" % n] for n in names], dtype=float)
            err = tgt - cur
            if np.abs(err).max() <= tol:
                return True
            if time.perf_counter() - t0 > timeout:
                return False
            goal = cur + np.clip(err, -step, step)
            robot.send_action({"%s.pos" % n: float(goal[i]) for i, n in enumerate(names)})
            time.sleep(dt)

    scores = []
    try:
        for trial in range(1, a.trials + 1):
            print("")
            print("=" * 60)
            print("trial %d/%d -- homing" % (trial, a.trials))
            ok = go_home()
            if not ok:
                print("  [!] homing did not fully settle; continuing anyway")

            if a.auto:
                # The homing move is itself the window for repositioning the block;
                # this is just a beat to get your hand clear before it takes over.
                for k in range(int(a.gap), 0, -1):
                    print("  starting in %d ..." % k)
                    time.sleep(1.0)
            else:
                print("  >>> place the block, then press Enter to start (%.0f s) <<<" % a.seconds)
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    print("stopped.")
                    break

            policy.reset()
            t0 = time.perf_counter()
            n = 0
            while time.perf_counter() - t0 < a.seconds:
                loop = time.perf_counter()
                obs = robot.get_observation()
                with torch.no_grad():
                    act = post(policy.select_action(pre(obs_batch(obs))))[0].cpu().numpy()
                robot.send_action({"%s.pos" % nm: float(act[i]) for i, nm in enumerate(names)})
                n += 1
                rest = dt - (time.perf_counter() - loop)
                if rest > 0:
                    time.sleep(rest)
            print("  done: %d steps in %.1f s (%.1f fps)"
                  % (n, time.perf_counter() - t0, n / (time.perf_counter() - t0)))

            if a.no_score or a.auto:
                continue
            while True:
                print("  score? %s" % STAGES)
                try:
                    v = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    v = ""
                if v in VALID:
                    scores.append(VALID[v])
                    break
                if v == "":
                    print("  (skipped)")
                    break
                print("  not a valid score")
    finally:
        print("")
        print("returning home before disconnect")
        try:
            go_home()
        except Exception:
            pass
        robot.disconnect()

    if scores:
        s = np.array(scores)
        print("")
        print("=" * 60)
        print("trials scored : %d" % len(s))
        print("mean score    : %.2f" % s.mean())
        print("reached  >=0.2: %d/%d" % ((s >= 0.2).sum(), len(s)))
        print("grasped  >=0.4: %d/%d" % ((s >= 0.4).sum(), len(s)))
        print("in bowl  ==1.0: %d/%d" % ((s >= 1.0).sum(), len(s)))
        print("scores        : %s" % ", ".join("%.1f" % x for x in s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
