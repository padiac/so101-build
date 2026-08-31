#!/usr/bin/env python3
"""
同时实时监视主臂和从臂的关节位置 —— 纯读取，不驱动电机。

用途一：确认主从方向是否一致。
    把两条臂的**同一个关节朝同一个物理方向**搬，看两边数字是不是**同向**变化。
    一个增一个减 = 方向相反，遥操作会反着动，需要在标定文件里把该关节的
    drive_mode 改成 1。

用途二：标定前确认所有关节都落在 0..4094（不在编码器零点附近）。

用法：
    python watch_both.py
    python watch_both.py --leader-port /dev/ttyACM0 --follower-port /dev/ttyACM1
"""

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leader-port", default="/dev/ttyACM0")
    ap.add_argument("--follower-port", default="/dev/ttyACM1")
    a = ap.parse_args()

    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    lead = SO101Leader(SO101LeaderConfig(port=a.leader_port, id="watch_l"))
    foll = SO101Follower(SO101FollowerConfig(port=a.follower_port, id="watch_f"))
    lead.bus.connect()
    foll.bus.connect()

    names = list(lead.bus.motors)
    prev = {}

    print("双臂实时监视。把同一个关节朝同一个物理方向搬，比较两边的箭头。")
    print("  箭头同向 = 方向一致，OK")
    print("  箭头反向 = 该关节主从相反，遥操作会反着动")
    print("  ! 表示读数超出 0-4094，标定会失败，要把该关节搬回区间内")
    print("Ctrl+C 退出")
    print("")

    def arrow(name, side, val):
        key = (side, name)
        old = prev.get(key)
        prev[key] = val
        if old is None:
            return " "
        d = val - old
        if d > 3:
            return "+"
        if d < -3:
            return "-"
        return " "

    try:
        while True:
            print(chr(27) + "[H" + chr(27) + "[J", end="")
            print("{:<15}{:>12}{:>4}   {:>12}{:>4}   {}".format(
                "joint", "LEADER", "", "FOLLOWER", "", "方向"))
            print("-" * 62)
            for n in names:
                try:
                    lp = lead.bus.read("Present_Position", n, normalize=False)
                except Exception:
                    lp = None
                try:
                    fp = foll.bus.read("Present_Position", n, normalize=False)
                except Exception:
                    fp = None

                if lp is None or fp is None:
                    print("{:<15}   读取失败".format(n))
                    continue

                la = arrow(n, "l", lp)
                fa = arrow(n, "f", fp)
                lbad = "!" if not (0 <= lp <= 4094) else " "
                fbad = "!" if not (0 <= fp <= 4094) else " "

                if la in "+-" and fa in "+-":
                    verdict = "一致" if la == fa else "*** 相反 ***"
                else:
                    verdict = ""

                print("{:<15}{:>12}{}{}   {:>12}{}{}   {}".format(
                    n, lp, lbad, la, fp, fbad, fa, verdict))
            print("")
            print("Ctrl+C 退出")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("")
    finally:
        lead.bus.disconnect()
        foll.bus.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
