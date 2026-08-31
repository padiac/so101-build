#!/usr/bin/env python3
"""
串口通信可靠性探针 —— 连续做大量小的读/写往返，统计失败率。

用来区分：
  * 偶发超时（失败率低，重试能救）
  * 系统性问题（某颗舵机总失败，或失败率很高 -> 串口延迟计时器/驱动问题）

用法：
    python probe_serial.py --port COM8 --rounds 30
    python probe_serial.py --port /dev/ttyACM1 --rounds 30
"""

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--arm", choices=["follower", "leader"], default="follower")
    ap.add_argument("--rounds", type=int, default=30)
    a = ap.parse_args()

    if a.arm == "follower":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        dev = SO101Follower(SO101FollowerConfig(port=a.port, id="probe"))
    else:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        dev = SO101Leader(SO101LeaderConfig(port=a.port, id="probe"))

    bus = dev.bus
    bus.connect()
    names = list(bus.motors)

    stats = {n: {"read_ok": 0, "read_err": 0, "write_ok": 0, "write_err": 0} for n in names}
    lat = []

    print("端口 {}   {} 轮 x {} 颗舵机，每轮 1 读 + 1 写".format(a.port, a.rounds, len(names)))
    print("")

    for r in range(a.rounds):
        for n in names:
            t0 = time.perf_counter()
            try:
                bus.read("Present_Position", n, normalize=False)
                stats[n]["read_ok"] += 1
            except Exception:
                stats[n]["read_err"] += 1
            lat.append((time.perf_counter() - t0) * 1000)

            try:
                # 写一个无害的值：把 Lock 写成它当前的值
                cur = bus.read("Lock", n)
                bus.write("Lock", n, cur)
                stats[n]["write_ok"] += 1
            except Exception:
                stats[n]["write_err"] += 1
        sys.stdout.write("\r第 {}/{} 轮".format(r + 1, a.rounds))
        sys.stdout.flush()

    print("")
    print("")
    print("{:<15}{:>10}{:>10}{:>11}{:>11}".format("motor", "读成功", "读失败", "写成功", "写失败"))
    print("-" * 58)
    tot_err = 0
    for n in names:
        s = stats[n]
        tot_err += s["read_err"] + s["write_err"]
        flag = "   <-- 有失败" if (s["read_err"] or s["write_err"]) else ""
        print("{:<15}{:>10}{:>10}{:>11}{:>11}{}".format(
            n, s["read_ok"], s["read_err"], s["write_ok"], s["write_err"], flag))

    lat.sort()
    n_ = len(lat)
    print("")
    print("单次读往返延迟:  中位 {:.2f}ms   p95 {:.2f}ms   最大 {:.2f}ms".format(
        lat[n_ // 2], lat[int(n_ * 0.95)], lat[-1]))
    print("总失败 {} 次 / {} 次操作  =  {:.2f}%".format(
        tot_err, a.rounds * len(names) * 2, 100.0 * tot_err / (a.rounds * len(names) * 2)))
    print("")
    if lat[n_ // 2] > 8:
        print("[!] 中位延迟 >8ms —— 典型的串口**延迟计时器**问题。")
        print("   设备管理器 -> 端口 -> 属性 -> 端口设置 -> 高级 -> 延迟计时器 改成 1ms")
    bus.disconnect(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
