#!/usr/bin/env python3
"""
读取一条臂上每颗舵机的原始位置和 Homing_Offset —— 纯读取，不驱动电机。

标定报 "Magnitude NNNN exceeds 2047" 时用这个定位问题舵机。

原理（Feetech）：
    Present_Position = Actual_Position - Homing_Offset
    标定要写的 Homing_Offset = Present_Position - 2047
    Homing_Offset 是 11 位符号-幅值寄存器，上限 ±2047

=> 只要 Present_Position 落在 0..4094 之内，算出的 offset 必然合法。
   报错说明某颗舵机的读数跑到了这个区间之外（转过了编码器零点）。
   解法是**用手把那个关节搬回区间内**，不需要拆任何东西。

用法：
    python diag_motors.py leader
    python diag_motors.py follower --port /dev/ttyACM1
    python diag_motors.py leader --watch     # 实时刷新，边搬边看
    python diag_motors.py leader --reset     # 清零所有 Homing_Offset
"""

import argparse
import sys
import time

NL = chr(10)


def build(arm, port):
    if arm == "follower":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        return SO101Follower(SO101FollowerConfig(port=port, id="diag"))
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    return SO101Leader(SO101LeaderConfig(port=port, id="diag"))


def dump(bus, title):
    print("")
    print(title)
    hdr = "{:<15}{:>4}{:>10}{:>10}{:>12}   {}".format(
        "motor", "id", "Present", "Homing", "P-2047", "状态")
    print(hdr)
    print("-" * 62)
    bad = []
    for m in bus.motors:
        mid = bus.motors[m].id
        try:
            p = bus.read("Present_Position", m, normalize=False)
            h = bus.read("Homing_Offset", m)
        except Exception as e:
            print("{:<15}{:>4}   读取失败: {}".format(m, mid, e))
            bad.append(m)
            continue
        off = p - 2047
        if not (0 <= p <= 4094):
            state = "!! 超出 0-4094，搬这个关节"
            bad.append(m)
        else:
            state = "OK"
        print("{:<15}{:>4}{:>10}{:>10}{:>12}   {}".format(m, mid, p, h, off, state))
    return bad


def watch(bus):
    print("实时监视。用手搬关节，看数字变化。")
    print("目标：每颗都落在 0-4094 之内（最好 1000-3000，留余量）")
    print("Ctrl+C 退出")
    print("")
    try:
        while True:
            cells = []
            allok = True
            for m in bus.motors:
                try:
                    p = bus.read("Present_Position", m, normalize=False)
                except Exception:
                    cells.append("{}=???".format(m[:5]))
                    allok = False
                    continue
                bad = not (0 <= p <= 4094)
                if bad:
                    allok = False
                cells.append("{}={}{}".format(m[:5], p, "!" if bad else ""))
            mark = "   <== 全部就绪，可以标定" if allok else ""
            sys.stdout.write(chr(13) + "  ".join(cells) + mark + "      ")
            sys.stdout.flush()
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("")
        print("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["leader", "follower"])
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--watch", action="store_true", help="实时刷新，边搬边看")
    ap.add_argument("--reset", action="store_true", help="清零所有 Homing_Offset")
    a = ap.parse_args()

    dev = build(a.arm, a.port)
    bus = dev.bus
    bus.connect()

    if a.watch:
        watch(bus)
        bus.disconnect()
        return 0

    bad = dump(bus, "=== 当前状态 ===")

    if a.reset:
        print("")
        print("=== 清零 Homing_Offset ===")
        for m in bus.motors:
            bus.write("Homing_Offset", m, 0)
            print("  {} -> 0".format(m))
        bad = dump(bus, "=== 清零后 ===")

    print("")
    if bad:
        print("需要处理: {}".format(", ".join(bad)))
        print("用手把这些关节搬进 0-4094 区间，可以配合实时监视：")
        print("  python diag_motors.py {} --port {} --watch".format(a.arm, a.port))
    else:
        print("全部正常，可以跑标定。")

    bus.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
