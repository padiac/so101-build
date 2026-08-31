#!/usr/bin/env python3
"""
USB 设备掉线监视器 —— 找出到底是哪根线/哪个口不稳。

持续轮询串口和摄像头，出现/消失都打时间戳。
跑起来之后**逐根轻轻晃动线缆和接头**，看哪一根会触发掉线。

用法：
    python watch_usb_win.py
    python watch_usb_win.py --seconds 300
"""

import argparse
import sys
import time

ARM = {"5B3E090575": "从臂 follower", "5B61033038": "主臂 leader"}


def snapshot():
    from serial.tools import list_ports
    ports = {}
    for p in list_ports.comports():
        if p.serial_number:                       # 只看有 SN 的，滤掉蓝牙/COM1
            ports[p.serial_number] = p.device
    cams = 0
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_PnPEntity | "
             "Where-Object { $_.PNPClass -eq 'Camera' }).Count"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        cams = int(out) if out.isdigit() else -1
    except Exception:
        cams = -1
    return ports, cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--interval", type=float, default=0.5)
    a = ap.parse_args()

    prev, prev_cams = snapshot()
    print("基线：")
    for sn, dev in sorted(prev.items()):
        print("   %-14s %-6s %s" % (sn, dev, ARM.get(sn, "")))
    print("   摄像头 %d 个" % prev_cams)
    print("")
    print("开始监视 %ds —— 现在**逐根轻轻晃动线缆和接头**，看哪根触发掉线。" % a.seconds)
    print("Ctrl+C 提前结束。")
    print("")

    t0 = time.time()
    events = 0
    try:
        while time.time() - t0 < a.seconds:
            cur, cams = snapshot()
            t = time.strftime("%H:%M:%S")
            for sn in prev:
                if sn not in cur:
                    events += 1
                    print("[%s] [X] 掉线  %s  %s  (%s)" % (
                        t, sn, prev[sn], ARM.get(sn, "")))
            for sn in cur:
                if sn not in prev:
                    events += 1
                    print("[%s] [OK] 回来  %s  %s  (%s)" % (
                        t, sn, cur[sn], ARM.get(sn, "")))
                elif cur[sn] != prev[sn]:
                    events += 1
                    print("[%s] [!] COM 变号 %s: %s -> %s  (%s)" % (
                        t, sn, prev[sn], cur[sn], ARM.get(sn, "")))
            if cams != prev_cams:
                events += 1
                print("[%s] [CAM] 摄像头数量 %d -> %d" % (t, prev_cams, cams))
            prev, prev_cams = cur, cams
            sys.stdout.write("\r  监视中 %ds ... 事件 %d   " % (int(time.time() - t0), events))
            sys.stdout.flush()
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass

    print("")
    print("")
    print("共 %d 个事件。" % events)
    if events == 0:
        print("晃动过程中没有掉线 —— 问题可能只在负载下出现，")
        print("那就一边跑 .\robot.ps1 teleop 一边观察。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
