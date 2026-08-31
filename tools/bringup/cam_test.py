#!/usr/bin/env python3
"""
摄像头可用性测试 —— Windows / Linux 通用。

回答：这台机器上有几个摄像头？能不能抓帧？分辨率和帧率是多少？

用法：
    python cam_test.py                 # 扫描 index 0..5
    python cam_test.py --max-index 10
    python cam_test.py --index 0 --save   # 抓一帧存成图片
"""

import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=6)
    ap.add_argument("--index", type=int, default=None, help="只测这一个")
    ap.add_argument("--save", action="store_true", help="保存一帧到 logs/")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    import cv2

    # Windows 上 DSHOW 后端比默认的 MSMF 更稳、开得更快
    backends = []
    if sys.platform.startswith("win"):
        backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    else:
        backends = [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]

    idxs = [a.index] if a.index is not None else list(range(a.max_index))
    found = []

    for be_name, be in backends:
        print("")
        print("=== 后端 {} ===".format(be_name))
        for i in idxs:
            cap = cv2.VideoCapture(i, be)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
            # 预热：刚打开时前若干帧常常是黑的（自动曝光还没收敛）
            frame = None
            ok = False
            for _ in range(20):
                ok, frame = cap.read()
                if ok and frame is not None:
                    time.sleep(0.03)
            if not ok or frame is None:
                print("  index {}: 打开了但读不到帧".format(i))
                cap.release()
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            cc = "".join(chr((fourcc >> (8 * k)) & 0xFF) for k in range(4))

            # 实测帧率
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 1.0:
                if cap.read()[0]:
                    n += 1
            real = n / (time.perf_counter() - t0)

            bright = float(frame.mean())
            note = ""
            if bright < 5:
                note = "   <-- 画面全黑！检查隐私挡片/环境光"
            elif bright < 20:
                note = "   <-- 很暗"
            print("  index {}: {}x{}  报告fps={:.0f}  实测fps={:.1f}  格式={}  亮度={:.1f}{}".format(
                i, w, h, fps, real, cc, bright, note))
            found.append((be_name, i, w, h, real, cc))

            if a.save:
                os.makedirs("logs", exist_ok=True)
                p = "logs/cam_{}_{}.jpg".format(be_name, i)
                cv2.imwrite(p, frame)
                print("        存图 {}".format(p))
            cap.release()

    print("")
    if found:
        print("找到 {} 个可用摄像头。".format(len(found)))
        print("")
        print("注意：")
        print("  * 实测fps 明显低于报告fps = 带宽或曝光问题，多相机时会更明显")
        print("  * 格式 MJPG 比 YUY2 省带宽，多相机务必用 MJPG")
        print("  * 采数据前记得关掉自动对焦和自动曝光")
    else:
        print("没找到可用摄像头。")
        if not sys.platform.startswith("win"):
            print("在 WSL 里这是预期结果 —— WSL 内核没有 UVC 支持。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
