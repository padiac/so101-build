#!/usr/bin/env python3
r"""
相机索引管理 —— 防止顶部/手腕视角互换导致数据集静默损坏。

为什么需要这个：
    两个 U20CAM-1080P 电气上完全相同，**没有序列号**：
        USB\VID_0C45&PID_6366&MI_00\7&29CD3151&0&0000
        USB\VID_0C45&PID_6366&MI_00\6&183AF011&0&0000
    后面那串是 USB 端口路径派生的实例 ID，不是序列号。
    OpenCV / DirectShow / pygrabber 都只给索引和一个相同的名字，无法区分。

    而索引顺序会随插拔、开机顺序变化。**一旦顶部和手腕互换，
    策略学到的映射全错，而且训练能正常跑完，你不会立刻发现。**

解法：存一张参考图，每次采数据前比对，自动判定并在可疑时报警。
      这两个视角差异极大，比对非常可靠。

用法：
    python cams.py list                    # 抓图 + 存联系表，看清哪个是哪个
    python cams.py set --top 1 --wrist 0   # 记下映射 + 存参考图
    python cams.py verify                  # 采数据前跑这个
"""

import argparse
import json
import os
import sys
import time
# 不要强制 stdout 用 utf-8 —— Windows 控制台是 GBK，强制 utf-8 只会输出乱码。
# GBK 能表示中文，编不了的只有 emoji。规则：打印内容用 [OK]/[!]/[X] 这类 ASCII 标记。

MAP_FILE = "camera_map.json"
REF_DIR = "camera_refs"


def grab(idx, width=640, height=480, warm=20):
    import cv2
    # MUST match the backend lerobot opens cameras with, otherwise the index
    # numbers refer to different physical devices. DirectShow and Media
    # Foundation enumerate independently: resolve identified index 0 as the
    # overhead camera through DSHOW while lerobot opened index 0 through MSMF
    # and got a different camera. robot.ps1 pins the camera config to MSMF
    # (1400) for the same reason.
    be = cv2.CAP_MSMF if sys.platform.startswith("win") else cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, be)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = None
    for _ in range(warm):          # 预热：前几十帧是黑的，自动曝光没收敛
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.02)
    cap.release()
    return frame


def all_indices(maxi=4):
    out = []
    for i in range(maxi):
        f = grab(i, warm=8)
        if f is not None:
            out.append((i, f))
    return out


def cmd_list(a):
    import cv2
    import numpy as np
    got = all_indices(a.max_index)
    if not got:
        print("没找到相机")
        return 1
    os.makedirs("logs", exist_ok=True)
    tiles = []
    for i, f in got:
        t = f.copy()
        cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), (0, 255, 0), 3)
        cv2.putText(t, "index %d" % i, (14, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 0, 0), 6)
        cv2.putText(t, "index %d" % i, (14, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 255, 0), 2)
        tiles.append(t)
        print("index %d  亮度 %.1f" % (i, float(f.mean())))
    sheet = np.hstack(tiles)
    p = "logs/cams_contact_sheet.jpg"
    cv2.imwrite(p, sheet)
    print("\n联系表: %s" % p)
    print("看清楚之后： python cams.py set --top N --wrist M")
    return 0


def cmd_set(a):
    import cv2
    m = {"top": a.top, "wrist": a.wrist}
    os.makedirs(REF_DIR, exist_ok=True)
    for role, idx in m.items():
        f = grab(idx)
        if f is None:
            print("抓不到 index %d" % idx)
            return 1
        cv2.imwrite(os.path.join(REF_DIR, role + ".jpg"), f)
        print("index %d -> %-6s  参考图已存" % (idx, role))
    json.dump(m, open(MAP_FILE, "w"), indent=2)
    print("\n映射写入 %s" % MAP_FILE)
    return 0


def cmd_verify(a):
    """用【运动】区分，而不是比对参考图。

    为什么不用参考图：手腕相机装在臂上、顶部相机固定，两者画面差异极大，
    所以第一版用"和参考图比像素差"来判定。实测失败 —— 采数据时场景本来就
    一直在变（物体挪位、光照、臂的姿态），对自己的差值从 1.8 涨到 91，
    直接误报。**那个方法测的是"画面变没变"，不是"相机换没换"。**

    改用物理判据：**手腕相机跟着臂动，顶部相机不动。**

    [!] 第一版这里也错了：写成"动主臂"。但手腕相机装在**从臂**上，
    而动主臂只有在遥操作运行时才会带动从臂。结果测到的是"你的手在
    顶部画面里晃"（top 8.6% / wrist 0.1%）。**必须动从臂本身。**

    让从臂动一下，比较前后两帧变化的像素比例：
        手腕 -> 整幅画面都在变（相机自身在移动）
        顶部 -> 只有臂占的那一小块在变
    这个判据与场景内容无关。
    """
    import cv2
    import numpy as np
    if not os.path.exists(MAP_FILE):
        print("还没设置映射，先跑: python cams.py list")
        return 1
    m = json.load(open(MAP_FILE))

    def changed_fraction(idx, thresh=18):
        before = grab(idx)
        if before is None:
            return None
        input("  >>> 现在【用手把从臂明显地动一下】，然后回车 ...")
        after = grab(idx, warm=6)
        if after is None:
            return None
        a_ = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)
        b_ = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.int16)
        return float((np.abs(a_ - b_) > thresh).mean())

    # 从臂 1/345 减速比本来就难反驱，扭矩没关的话根本搬不动
    if a.follower_port:
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            dev = SO101Follower(SO101FollowerConfig(port=a.follower_port, id="camverify"))
            dev.bus.connect(handshake=False)
            dev.bus.disable_torque()
            dev.bus.disconnect(False)
            print("已关闭从臂扭矩，现在可以用手搬动。")
        except Exception as e:
            print("关扭矩失败（%s），如果搬不动就先跑一次 robot.ps1 health" % type(e).__name__)
    print("")
    print("接下来分别测两个相机。每次提示后，【用手把从臂明显地动一下】。")
    print("手腕相机装在从臂上，所以只有从臂动它才会动 —— 动主臂没用。")
    print("")
    frac = {}
    for role, idx in m.items():
        print("测 %s (index %d):" % (role, idx))
        f = changed_fraction(idx)
        if f is None:
            print("  抓不到 index %d" % idx)
            return 1
        frac[role] = f
        print("  变化像素占比 %.1f%%" % (f * 100))
        print("")

    print("%-8s%-8s%14s" % ("角色", "索引", "变化占比"))
    print("-" * 34)
    for role in m:
        print("%-8s%-8d%13.1f%%" % (role, m[role], frac[role] * 100))
    print("")

    if "wrist" not in frac or "top" not in frac:
        print("映射里缺 wrist 或 top")
        return 1

    w, t = frac["wrist"], frac["top"]
    if w > t * 1.8 and w > 0.15:
        print("[OK] wrist 的变化明显大于 top —— 映射正确，可以采数据。")
        return 0
    if t > w * 1.8 and t > 0.15:
        print("[!!] 反了！标成 top 的那个才是手腕相机。")
        print("     跑: python cams.py set --top %d --wrist %d" % (m["wrist"], m["top"]))
        return 2
    print("[??] 两边变化差不多，判定不了。可能是：")
    print("     - 臂动得太小，再试一次并【动大一点】")
    print("     - 手腕相机没装在臂上")
    print("     用 python cams.py list 肉眼确认。")
    return 3


ROBOT_CAM_NAME = "Innomaker"     # 机械臂用的那两个相机的设备名关键字


def device_names():
    """DirectShow 设备名列表，下标就是 OpenCV 的索引。"""
    from pygrabber.dshow_graph import FilterGraph
    return FilterGraph().get_input_devices()


def signature(bgr):
    """32x32 灰度、去均值、单位化。比的是视角构图，不是场景内容。"""
    import cv2
    import numpy as np
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32)).astype(np.float64)
    v = g - g.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).ravel()


def cmd_resolve(a):
    """Find the robot's two cameras empirically, by what they show.

    Earlier versions of this matched on the DirectShow device name and used that
    listing's position as the OpenCV index. That assumption is wrong: pygrabber
    enumerates through DirectShow while OpenCV may open through MSMF, and the
    two orderings do not have to agree. It picked an index that never delivered
    a frame.

    So trust nothing but the frames. Open every index, throw away the ones that
    fail to open or return black, and match what remains against the stored
    reference views. A camera that cannot produce a picture cannot be the answer,
    whatever it is called and wherever it sits in a device list.

    Refuses to write a map it cannot determine confidently -- if top and wrist
    are ever swapped, training still runs to completion and the policy silently
    learns the wrong mapping.
    """
    import cv2
    import numpy as np

    print("scanning indices 0..%d" % (a.max_index - 1))
    live = {}
    for i in range(a.max_index):
        f = grab(i, warm=12)
        if f is None:
            print("  index %d  cannot open" % i)
            continue
        m = float(f.mean())
        if m < 5.0:
            print("  index %d  BLACK (mean %.1f) -- ignored" % (i, m))
            continue
        print("  index %d  ok (mean %.1f)" % (i, m))
        live[i] = f
    print("")

    if len(live) < 2:
        print("[X] need at least 2 working cameras, found %d." % len(live))
        print("    Check the USB connections.")
        return 1

    refs = {}
    for role in ("top", "wrist"):
        f = os.path.join(REF_DIR, role + ".jpg")
        if not os.path.exists(f):
            print("[X] missing reference %s. Run once: python cams.py set --top N --wrist M" % f)
            return 1
        refs[role] = signature(cv2.imread(f))

    sig = {i: signature(f) for i, f in live.items()}

    print("%-8s %10s %10s" % ("index", "vs top", "vs wrist"))
    for i in sorted(sig):
        print("%-8d %10.3f %10.3f"
              % (i, float(sig[i] @ refs["top"]), float(sig[i] @ refs["wrist"])))
    print("")

    # Match on the TOP camera only, and take whatever is left as the wrist.
    #
    # Matching both was fragile: the wrist camera is bolted to the arm, so its
    # view is whatever the arm happens to be pointing at, and a reference shot
    # taken at one pose stops matching once the arm has moved. That is what
    # failed here -- wrist correlations of 0.09 and -0.36 against its own
    # reference, and the assignment came out ambiguous.
    #
    # The overhead camera is fixed. Its view changes only as much as the scene
    # does, which makes it the reliable anchor. There are exactly two working
    # robot cameras, so identifying one identifies the other.
    top_scores = {i: float(sig[i] @ refs["top"]) for i in sig}
    ranked = sorted(top_scores, key=lambda i: -top_scores[i])
    it = ranked[0]
    margin = top_scores[ranked[0]] - top_scores[ranked[1]]
    others = [i for i in sig if i != it]
    if len(others) != 1:
        print("[X] expected exactly one remaining camera, got %d: %s" % (len(others), others))
        print("    Unplug anything that is not the robot's two cameras, or check by eye:")
        print("    python cams.py list")
        return 2
    iw = others[0]

    print("top match: index %d (%.3f), next best %.3f, margin %.3f"
          % (it, top_scores[ranked[0]], top_scores[ranked[1]], margin))
    print("wrist: index %d (the remaining camera)" % iw)
    print("")
    if margin < 0.15:
        print("[X] the overhead view does not stand out clearly, map not written.")
        print("    The scene may have changed a lot since the reference was taken.")
        print("    Check by eye and re-store: python cams.py list")
        print("                               python cams.py set --top N --wrist M")
        return 2

    json.dump({"top": it, "wrist": iw}, open(MAP_FILE, "w"), indent=2)
    print("[OK] top=index %d   wrist=index %d   written to %s" % (it, iw, MAP_FILE))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list");   p.add_argument("--max-index", type=int, default=4)
    p = sub.add_parser("set")
    p.add_argument("--top", type=int, required=True)
    p.add_argument("--wrist", type=int, required=True)
    p = sub.add_parser("resolve")
    p.add_argument("--max-index", type=int, default=6)
    p = sub.add_parser("verify")
    p.add_argument("--follower-port", default=None,
                   help="从臂串口，用来先关掉扭矩（例如 COM8）")
    a = ap.parse_args()
    return {"list": cmd_list, "set": cmd_set,
            "resolve": cmd_resolve, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
