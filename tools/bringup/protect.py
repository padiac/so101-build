#!/usr/bin/env python3
"""
读取（可选修改）从臂舵机的过载保护相关寄存器。

shoulder_lift 反复报 Overload error 时用这个看是阈值太保守还是负载真的超标。

用法：
    python protect.py                      # 只读
    python protect.py --boost shoulder_lift  # 提高该关节的保护阈值
"""

import argparse
import sys

REGS = [
    "Max_Torque_Limit",       # 最大输出扭矩 0-1000
    "Torque_Limit",           # 当前扭矩上限
    "Protective_Torque",      # 触发保护后维持的扭矩
    "Overload_Torque",        # 过载判定阈值 (%)
    "Protection_Time",        # 过载持续多久才跳闸
    "Protection_Current",     # 电流保护阈值
    "Over_Current_Protection_Time",
    "Unloading_Condition",    # 哪些条件会卸载扭矩
    "LED_Alarm_Condition",    # 哪些条件点亮报警灯
    "Max_Temperature_Limit",
    "Min_Voltage_Limit",
    "Max_Voltage_Limit",
]

LIVE = ["Present_Load", "Present_Current", "Present_Temperature",
        "Present_Voltage", "Status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--boost", default=None,
                    help="要提高保护阈值的关节名，如 shoulder_lift")
    a = ap.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    dev = SO101Follower(SO101FollowerConfig(port=a.port, id="prot"))
    bus = dev.bus
    bus.connect()

    names = list(bus.motors)

    print("=" * 92)
    print("保护设置")
    print("{:<32}".format("register") + "".join("{:>10}".format(n[:9]) for n in names))
    print("-" * 92)
    for r in REGS:
        row = "{:<32}".format(r)
        for n in names:
            try:
                row += "{:>10}".format(bus.read(r, n))
            except Exception:
                row += "{:>10}".format("-")
        print(row)

    print("")
    print("=" * 92)
    print("实时状态")
    print("{:<32}".format("register") + "".join("{:>10}".format(n[:9]) for n in names))
    print("-" * 92)
    for r in LIVE:
        row = "{:<32}".format(r)
        for n in names:
            try:
                row += "{:>10}".format(bus.read(r, n))
            except Exception:
                row += "{:>10}".format("-")
        print(row)

    if a.boost:
        if a.boost not in names:
            print("未知关节: {}".format(a.boost))
            bus.disconnect()
            return 1
        print("")
        print("=== 提高 {} 的保护阈值 ===".format(a.boost))
        newvals = {
            "Max_Torque_Limit": 1000,   # 满扭矩
            "Protection_Current": 500,  # 电流保护放宽
            "Overload_Torque": 80,      # 过载判定放宽到 80%
            "Protection_Time": 200,     # 允许更长的持续高负载
        }
        for k, v in newvals.items():
            try:
                old = bus.read(k, a.boost)
                bus.write(k, a.boost, v)
                print("  {:<24} {} -> {}".format(k, old, v))
            except Exception as e:
                print("  {:<24} 失败: {}".format(k, e))
        print("")
        print("已写入 EEPROM，断电也保留。")

    bus.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
