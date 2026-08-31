"""把实时画面跟训练数据里的画面比对，判断 top/wrist 索引有没有换位。

为什么这么查：
    两个相机没有序列号，Windows 的 index 会在重插/重启后互换。
    换位后网络收到的是镜像输入，表现是"乱动、不去夹"，
    看起来像训练失败，其实是输入接错了。

    比对基准不是"现在哪个画面更亮"这类启发式，而是**训练时真实存下来的帧**。
    场景会变（物体挪了），但视角不会变 —— 手腕特写和俯视全景差异极大，
    实测分离度 47 倍，足够判定。

用法：
    python check_cam_match.py
"""

import sys

import cv2
import numpy as np


def sig(img):
    """粗粒度签名：灰度 + 缩到 32x32 + 去均值归一化。

    缩到这么小是故意的 —— 我们要比的是"视角/构图"，不是"桌上摆了什么"。
    保留细节反而会被场景变化干扰。
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32)).astype(np.float64)
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 1e-9 else g


def corr(a, b):
    return float((a * b).sum())


def train_frames(root, key, n=5):
    """从训练数据的视频里取几帧。"""
    import glob
    pat = f"{root}/videos/observation.images.{key}/**/*.mp4"
    files = sorted(glob.glob(pat, recursive=True))
    if not files:
        raise SystemExit(f"no video for {key} under {pat}")
    cap = cv2.VideoCapture(files[0])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
    out = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(sig(fr))
    cap.release()
    if not out:
        raise SystemExit(f"could not decode {files[0]}")
    return out


def live_frame(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fr = None
    for _ in range(10):          # 丢掉前几帧，等自动曝光稳定
        ok, f = cap.read()
        if ok:
            fr = f
    cap.release()
    return sig(fr) if fr is not None else None


def main():
    root = "datasets/so101_v3"
    ref = {k: train_frames(root, k) for k in ("top", "wrist")}
    print("training reference frames loaded: top=%d wrist=%d"
          % (len(ref["top"]), len(ref["wrist"])))

    live = {}
    for idx in (0, 1):
        s = live_frame(idx)
        if s is None:
            print("  index %d : cannot open" % idx)
        else:
            live[idx] = s
            print("  index %d : captured" % idx)
    if len(live) < 2:
        raise SystemExit("need both cameras")

    print()
    print("%-10s %10s %10s" % ("", "vs top", "vs wrist"))
    score = {}
    for idx, s in live.items():
        st = max(corr(s, r) for r in ref["top"])
        sw = max(corr(s, r) for r in ref["wrist"])
        score[idx] = (st, sw)
        print("index %-4d %10.3f %10.3f" % (idx, st, sw))

    # 两种分配方式，取总分高的
    a = score[0][0] + score[1][1]     # 0=top, 1=wrist
    b = score[0][1] + score[1][0]     # 0=wrist, 1=top
    print()
    print("assignment  0=top,1=wrist : %.3f" % a)
    print("assignment  0=wrist,1=top : %.3f" % b)
    best = {"top": 0, "wrist": 1} if a > b else {"top": 1, "wrist": 0}
    print()
    print("BEST: top=index %d  wrist=index %d   (margin %.3f)"
          % (best["top"], best["wrist"], abs(a - b)))

    import json
    cur = json.load(open("camera_map.json"))
    print("camera_map.json currently: top=%s wrist=%s" % (cur.get("top"), cur.get("wrist")))
    if cur.get("top") != best["top"] or cur.get("wrist") != best["wrist"]:
        print()
        print(">>> MISMATCH. camera_map.json does not match what training saw.")
        print(">>> Fix with: python cams.py set --top %d --wrist %d"
              % (best["top"], best["wrist"]))
        return 1
    print()
    print("[OK] mapping matches training data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
