#!/usr/bin/env python3
"""
带负载记录的遥操作 —— 用来抓 Overload 的现场。

和 lerobot-teleoperate 的区别：
  * 每个周期记录每颗舵机的 Present_Load / Present_Current / Status
  * 负载超过阈值就**主动停机并关扭矩**，不等舵机自己跳闸
  * 全程写 CSV，事后能画出扭矩爬升曲线
  * 退出时保证关扭矩（官方脚本在已跳闸时会崩在这一步）

用法：
    python teleop_diag.py                      # 默认限速 15, 30fps, 负载上限 700
    python teleop_diag.py --max-rel 20 --load-limit 600
    python teleop_diag.py --dry-run            # 只读不驱动，安全观察
"""

import argparse
import csv
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leader-port", default="/dev/ttyACM0")
    ap.add_argument("--follower-port", default="/dev/ttyACM1")
    ap.add_argument("--max-rel", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--load-limit", type=int, default=700,
                    help="Present_Load 绝对值超过这个就停机（满量程 1000）")
    ap.add_argument("--dry-run", action="store_true",
                    help="不发指令，只记录，用于安全观察")
    a = ap.parse_args()

    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    lead = SO101Leader(SO101LeaderConfig(port=a.leader_port, id="my_leader"))
    foll = SO101Follower(SO101FollowerConfig(
        port=a.follower_port, id="my_follower",
        max_relative_target=None if a.dry_run else a.max_rel))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = "logs/teleop_diag_{}.csv".format(stamp)

    lead.connect(calibrate=False)
    foll.connect(calibrate=False)
    names = list(foll.bus.motors)

    print("记录到 {}".format(path))
    print("负载上限 {}  限速 {}  fps {}  dry_run={}".format(
        a.load_limit, a.max_rel, a.fps, a.dry_run))
    print("Ctrl+C 停止。超限会自动停机。")
    print("")

    f = open(path, "w", newline="")
    w = csv.writer(f)
    head = ["t"]
    for n in names:
        head += [n + ".goal", n + ".pos", n + ".load", n + ".cur", n + ".status"]
    w.writerow(head)

    period = 1.0 / a.fps
    t0 = time.perf_counter()
    stop_reason = "user"

    try:
        while True:
            tick = time.perf_counter()
            action = lead.get_action()

            if not a.dry_run:
                foll.send_action(action)

            row = [round(tick - t0, 3)]
            worst = 0
            worst_name = ""
            for n in names:
                g = action.get(n + ".pos")
                try:
                    p = foll.bus.read("Present_Position", n)
                    ld = foll.bus.read("Present_Load", n)
                    cu = foll.bus.read("Present_Current", n)
                    st = foll.bus.read("Status", n)
                except Exception:
                    p = ld = cu = st = None
                row += [g, p, ld, cu, st]
                if isinstance(ld, int) and abs(ld) > worst:
                    worst, worst_name = abs(ld), n
                if isinstance(st, int) and st != 0:
                    stop_reason = "status {} on {}".format(st, n)
                    raise KeyboardInterrupt
            w.writerow(row)

            sys.stdout.write("\rt={:6.1f}s  峰值负载 {:>4} @ {:<14}".format(
                tick - t0, worst, worst_name))
            sys.stdout.flush()

            if worst > a.load_limit:
                stop_reason = "load {} on {} 超过上限".format(worst, worst_name)
                raise KeyboardInterrupt

            dt = period - (time.perf_counter() - tick)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    except Exception:
        stop_reason = "exception"
        traceback.print_exc()
    finally:
        print("")
        print("停止原因: {}".format(stop_reason))
        f.close()
        for dev, tag in ((foll, "follower"), (lead, "leader")):
            try:
                dev.bus.disable_torque()
            except Exception as e:
                print("  {} 关扭矩失败（可能已跳闸）: {}".format(tag, e))
            try:
                dev.bus.disconnect(False)
            except Exception:
                pass
        print("CSV: {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
