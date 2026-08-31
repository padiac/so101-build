#!/usr/bin/env python3
"""
给标定范围加安全余量 —— 防止遥操作把关节指令到机械死点上。

背景：
    标定"摇行程"那一步是用手把关节推到**机械死点**才记录 min/max 的。
    但遥操作会把主臂的极限位置映射成从臂的极限位置，于是从臂被指令去
    顶死点 -> 堵转 -> 持续满扭矩 -> Overload 跳闸。

    典型症状：机械完全灵活、单独驱动能轻松抬起（负载只有 10-20%）、
    但一跑遥操作某个关节立刻跳闸不动。

解法：把 range_min 抬高一点、range_max 压低一点，指令就永远够不到死点。
      wrist_roll 是 full-turn 关节（0-4095），跳过不动。

用法：
    python shrink_range.py --show                 # 只看当前值
    python shrink_range.py --margin 60            # 上下各收 60 counts
    python shrink_range.py --margin 60 --apply    # 真正写入（会先备份）
"""

import argparse
import json
import os
import shutil
import sys
import time

BASE = os.path.expanduser("~/.cache/huggingface/lerobot/calibration")
FILES = {
    "leader": os.path.join(BASE, "teleoperators/so_leader/my_leader.json"),
    "follower": os.path.join(BASE, "robots/so_follower/my_follower.json"),
}
SKIP = {"wrist_roll"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=int, default=60,
                    help="上下各收进多少 counts（4096 对应 360deg，60 约 5.3deg）")
    ap.add_argument("--show", action="store_true", help="只显示，不改")
    ap.add_argument("--apply", action="store_true", help="真正写入文件")
    a = ap.parse_args()

    for arm, path in FILES.items():
        if not os.path.exists(path):
            print("找不到 {}".format(path))
            continue
        d = json.load(open(path))
        print("=" * 74)
        print("{}   {}".format(arm.upper(), path))
        print("{:<15}{:>10}{:>10}   ->{:>10}{:>10}   {}".format(
            "joint", "min", "max", "min", "max", "span"))
        print("-" * 74)
        changed = False
        for k, v in d.items():
            lo, hi = v["range_min"], v["range_max"]
            if k in SKIP:
                print("{:<15}{:>10}{:>10}   ->{:>10}{:>10}   跳过(full-turn)".format(
                    k, lo, hi, lo, hi))
                continue
            nlo, nhi = lo + a.margin, hi - a.margin
            if nhi - nlo < 200:
                print("{:<15}{:>10}{:>10}   ->  余量过大，跳过".format(k, lo, hi))
                continue
            print("{:<15}{:>10}{:>10}   ->{:>10}{:>10}   {}".format(
                k, lo, hi, nlo, nhi, nhi - nlo))
            if not a.show:
                v["range_min"], v["range_max"] = nlo, nhi
                changed = True
        print("")

        if a.apply and changed:
            bak = path + ".bak_" + time.strftime("%Y%m%d_%H%M%S")
            shutil.copy2(path, bak)
            json.dump(d, open(path, "w"), indent=4)
            print("已写入。备份: {}".format(bak))
            print("")

    if not a.apply:
        print("以上是预览。确认没问题后加 --apply 写入：")
        print("  python shrink_range.py --margin {} --apply".format(a.margin))
        print("")
        print("写入后要重新把标定推进舵机 —— 直接跑遥操作即可，")
        print("connect() 时会把标定文件写进电机。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
