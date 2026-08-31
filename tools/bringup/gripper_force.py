#!/usr/bin/env python3
"""
调节夹爪握力（改 Max_Torque_Limit 等保护寄存器）。

原理：夹爪是**位置控制**，握力来自"指令闭合过头"产生的位置误差，
      大小被 Max_Torque_Limit 封顶。所以调这个值 = 调握力上限。

LeRobot 出厂对 gripper 已经限过一道：
    Max_Torque_Limit  500   (其他关节 1000)
    Protection_Current 250  (其他关节 310)
    Overload_Torque     25  (其他关节 80) -> 堵转后自动泄力到 25%

用法：
    python gripper_force.py                    # 只看当前值
    python gripper_force.py --level soft       # 轻柔（易碎物、打印件）
    python gripper_force.py --level normal     # LeRobot 默认
    python gripper_force.py --level firm       # 抓重物/滑物
    python gripper_force.py --torque 350       # 直接指定 0-1000

[!] 加大握力之前，先考虑给夹爪贴防滑垫 —— 摩擦力比握力更管用，
   而且不会挤坏东西。
"""

import argparse
import sys

LEVELS = {
    "soft":   {"Max_Torque_Limit": 300, "Protection_Current": 180, "Overload_Torque": 20},
    "normal": {"Max_Torque_Limit": 500, "Protection_Current": 250, "Overload_Torque": 25},
    "firm":   {"Max_Torque_Limit": 700, "Protection_Current": 320, "Overload_Torque": 35},
}
SHOW = ["Max_Torque_Limit", "Torque_Limit", "Protection_Current",
        "Overload_Torque", "Protective_Torque", "Protection_Time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--joint", default="gripper")
    ap.add_argument("--level", choices=sorted(LEVELS))
    ap.add_argument("--torque", type=int, help="直接指定 Max_Torque_Limit (0-1000)")
    a = ap.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    dev = SO101Follower(SO101FollowerConfig(port=a.port, id="grip"))
    bus = dev.bus
    bus.connect()

    def show(title):
        print("")
        print(title)
        for r in SHOW:
            try:
                print("  {:<22} {}".format(r, bus.read(r, a.joint)))
            except Exception as e:
                print("  {:<22} 读取失败 {}".format(r, e))

    show("=== 当前 ({}) ===".format(a.joint))

    new = None
    if a.level:
        new = dict(LEVELS[a.level])
    if a.torque is not None:
        new = new or {}
        new["Max_Torque_Limit"] = max(0, min(1000, a.torque))

    if new:
        print("")
        print("=== 写入 ===")
        for k, v in new.items():
            try:
                bus.write(k, a.joint, v)
                print("  {:<22} -> {}".format(k, v))
            except Exception as e:
                print("  {:<22} 失败 {}".format(k, e))
        show("=== 写入后 ===")
        print("")
        print("注意：LeRobot 每次 connect() 会把 gripper 重设成 500/250/25。")
        print("      要永久生效得改 so_follower.py 的 configure()，")
        print("      或者每次连接后重跑一次本脚本。")
    else:
        print("")
        print("加 --level soft|normal|firm 或 --torque N 来修改。")

    bus.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
