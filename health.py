#!/usr/bin/env python3
"""
读取舵机健康状态：错误标志 / 电压 / 温度 / 负载 / 电流 —— 纯读取。

出现 "Overload error!" 或者舵机闪灯时用这个定位。

用法：
    python health.py                     # 两条臂都查
    python health.py --leader-port /dev/ttyACM0 --follower-port /dev/ttyACM1
    python health.py --clear             # 尝试清除错误标志（需先消除过载原因）
"""

import argparse
import sys

# Feetech STS/SMS Status(65) 错误位
BITS = [
    (0x01, "过压/欠压"),
    (0x02, "角度超限"),
    (0x04, "过热"),
    (0x08, "电流异常"),
    (0x10, "角度传感器"),
    (0x20, "过载"),
]


def decode(st):
    if st == 0:
        return "正常"
    hits = [name for m, name in BITS if st & m]
    return "!! " + " + ".join(hits) if hits else "!! 未知位 0x{:02X}".format(st)


def check(dev, title):
    bus = dev.bus
    bus.connect()
    print("=" * 78)
    print(title)
    hdr = "{:<15}{:>3}{:>8}{:>7}{:>7}{:>9}{:>9}   {}".format(
        "motor", "id", "Status", "电压V", "温度C", "负载", "电流", "判断")
    print(hdr)
    print("-" * 78)
    bad = []
    for m in bus.motors:
        mid = bus.motors[m].id
        vals = {}
        for reg in ("Status", "Present_Voltage", "Present_Temperature",
                    "Present_Load", "Present_Current"):
            try:
                vals[reg] = bus.read(reg, m)
            except Exception:
                vals[reg] = None
        st = vals["Status"]
        v = vals["Present_Voltage"]
        volt = "{:.1f}".format(v / 10.0) if isinstance(v, int) else "?"
        verdict = decode(st) if isinstance(st, int) else "读取失败"
        if not (isinstance(st, int) and st == 0):
            bad.append(m)
        print("{:<15}{:>3}{:>8}{:>7}{:>7}{:>9}{:>9}   {}".format(
            m, mid,
            st if st is not None else "?",
            volt,
            vals["Present_Temperature"] if vals["Present_Temperature"] is not None else "?",
            vals["Present_Load"] if vals["Present_Load"] is not None else "?",
            vals["Present_Current"] if vals["Present_Current"] is not None else "?",
            verdict))
    print("")
    return bus, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leader-port", default="/dev/ttyACM0")
    ap.add_argument("--follower-port", default="/dev/ttyACM1")
    ap.add_argument("--only", choices=["leader", "follower"])
    ap.add_argument("--clear", action="store_true",
                    help="尝试关闭扭矩以清除错误（先把机械原因排除掉）")
    a = ap.parse_args()

    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    jobs = []
    if a.only != "follower":
        jobs.append(("LEADER (主臂)  " + a.leader_port,
                     SO101Leader(SO101LeaderConfig(port=a.leader_port, id="hl"))))
    if a.only != "leader":
        jobs.append(("FOLLOWER (从臂) " + a.follower_port,
                     SO101Follower(SO101FollowerConfig(port=a.follower_port, id="hf"))))

    allbad = []
    for title, dev in jobs:
        bus, bad = check(dev, title)
        allbad += [(title.split()[0], m) for m in bad]
        if a.clear and bad:
            print("尝试清除 {} 的错误...".format(title.split()[0]))
            for m in bad:
                try:
                    bus.write("Torque_Enable", m, 0)
                    print("  {} 扭矩已关闭".format(m))
                except Exception as e:
                    print("  {} 失败: {}".format(m, e))
            print("")
        bus.disconnect()

    if allbad:
        print("有错误标志的舵机:")
        for arm, m in allbad:
            print("  {} {}".format(arm, m))
        print("")
        print("过载错误的清除方法：**断电重启驱动板**（拔掉 DC 电源，等 5 秒，插回）。")
        print("断电前先用手把臂托低，别让它悬在半空。")
    else:
        print("全部正常。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
