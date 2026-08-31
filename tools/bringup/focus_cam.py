#!/usr/bin/env python3
"""
镜头调焦助手 —— 实时显示清晰度分数，一边拧镜头一边看数字。

M12 镜头出厂焦距随机，必须手动调。肉眼判断不准，用拉普拉斯方差
（Laplacian variance）作指标：值越大越清晰。中央区域权重更高。

用法：
    python focus_cam.py --index 1
    python focus_cam.py --index 1 --show          # 预览窗口（需完整版 opencv）
    python focus_cam.py --index 1 --snap          # 每秒存一张到 logs/focus_live.jpg
    python focus_cam.py --index 1 --mjpg          # 用 MJPG（多相机时省带宽）

注意：LeRobot 依赖的是 opencv-python-headless，**没有 GUI**，--show 会不可用。
      脚本会自动降级成纯数字模式，不会崩。要用窗口就装完整版：
          pip uninstall -y opencv-python-headless
          pip install opencv-python

调好之后：**在镜头螺纹接缝处点一滴指甲油或热熔胶固定**，
否则焦点会慢慢漂，而且你不会立刻发现。
"""

import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--show", action="store_true", help="预览窗口（需完整版 opencv）")
    ap.add_argument("--snap", action="store_true", help="每秒存一张实时图")
    ap.add_argument("--mjpg", action="store_true", help="用 MJPG 格式")
    a = ap.parse_args()

    import cv2

    be = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    cap = cv2.VideoCapture(a.index, be)
    if not cap.isOpened():
        print("打不开 index {}".format(a.index))
        return 1
    if a.mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(20):          # 预热，等自动曝光收敛
        cap.read()
        time.sleep(0.02)

    show = a.show
    if show:                      # 先探一下 GUI 到底能不能用
        try:
            cv2.namedWindow("focus", cv2.WINDOW_AUTOSIZE)
        except Exception:
            show = False
            print("这个 opencv 是 headless 版，没有 GUI —— 自动切成纯数字模式。")
            print("想要窗口就装完整版： pip uninstall -y opencv-python-headless "
                  "&& pip install opencv-python")
            print("或者加 --snap，每秒存一张图到 logs/focus_live.jpg\n")

    if a.snap:
        os.makedirs("logs", exist_ok=True)

    print("慢慢拧镜头，让分数最大化。把相机对准有细节的东西，距离用你实际的工作距离。")
    print("Ctrl+C 退出" + ("（窗口里按 q 也行）" if show else ""))
    print("")

    best = 0.0
    last_snap = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = g.shape
            full = cv2.Laplacian(g, cv2.CV_64F).var()
            cen = cv2.Laplacian(g[h // 3:2 * h // 3, w // 3:2 * w // 3], cv2.CV_64F).var()
            score = 0.4 * full + 0.6 * cen
            best = max(best, score)
            bar = int(min(score / max(best, 1e-6), 1.0) * 40)
            mark = "  <== 峰值" if score >= best * 0.995 else ""
            sys.stdout.write("\r清晰度 {:8.1f}  (峰值 {:8.1f})  [{}{}]{}   ".format(
                score, best, "#" * bar, "." * (40 - bar), mark))
            sys.stdout.flush()

            now = time.time()
            if a.snap and now - last_snap > 1.0:
                cv2.imwrite("logs/focus_live.jpg", frame)
                last_snap = now

            if show:
                disp = frame.copy()
                cv2.putText(disp, "focus {:.0f} / best {:.0f}".format(score, best),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(disp, (w // 3, h // 3), (2 * w // 3, 2 * h // 3), (0, 255, 0), 1)
                try:
                    cv2.imshow("focus", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception:
                    show = False
    except KeyboardInterrupt:
        pass
    finally:
        print("")
        print("峰值清晰度 {:.1f}".format(best))
        print("调好后用指甲油/热熔胶把镜头螺纹固定住。")
        cap.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
