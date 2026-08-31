#!/usr/bin/env python3
"""
受控抬升测试 —— 单独驱动一个关节，小步上抬，全程监控负载。

回答一个问题：这颗舵机能不能抬起这条臂？抬的时候负载曲线长什么样？

安全设计：
  * 一次只动**一个**关节，其余全部保持扭矩关闭
  * 每步只走 --step 度（默认 2°），走完停下来读负载
  * 负载超过 --load-limit 或 Status != 0 立刻停机关扭矩
  * 任何异常/Ctrl+C 都保证关扭矩

用法：
    python lift_test.py                          # 抬 shoulder_lift，2°一步，共 20 步
    python lift_test.py --joint elbow_flex
    python lift_test.py --step 1 --steps 30 --load-limit 600
    python lift_test.py --down                   # 反方向

[!] 跑之前：从臂夹紧、周围清空、手放在电源插头旁边。
"""

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--joint", default="shoulder_lift")
    ap.add_argument("--step", type=float, default=2.0, help="每步角度")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--load-limit", type=int, default=700, help="满量程 1000")
    ap.add_argument("--settle", type=float, default=0.4, help="每步后等待秒数")
    ap.add_argument("--down", action="store_true", help="反方向")
    a = ap.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    dev = SO101Follower(SO101FollowerConfig(
        port=a.port, id="my_follower", max_relative_target=None))
    dev.connect(calibrate=False)
    bus = dev.bus

    if a.joint not in bus.motors:
        print("未知关节: {}".format(a.joint))
        return 1

    # 只留目标关节有扭矩，其余全部松开
    bus.disable_torque()
    time.sleep(0.2)

    sign = -1.0 if a.down else 1.0
    start = bus.read("Present_Position", a.joint)
    lo = bus.read("Min_Position_Limit", a.joint)
    hi = bus.read("Max_Position_Limit", a.joint)

    print("关节 {}   起始 {:.1f}deg   raw限位 [{}, {}]".format(a.joint, start, lo, hi))
    print("每步 {}deg  共 {} 步  方向 {}".format(
        a.step, a.steps, "向下" if a.down else "向上"))
    print("负载上限 {}（满量程 1000）".format(a.load_limit))
    print("")
    print("{:>5}{:>10}{:>10}{:>9}{:>9}{:>8}".format(
        "step", "goal", "pos", "load", "current", "status"))
    print("-" * 52)

    reason = "完成全部步数"
    try:
        bus.enable_torque([a.joint])
        time.sleep(0.2)
        goal = start
        for i in range(1, a.steps + 1):
            goal = goal + sign * a.step
            bus.write("Goal_Position", a.joint, goal)
            time.sleep(a.settle)

            pos = bus.read("Present_Position", a.joint)
            ld = bus.read("Present_Load", a.joint)
            cu = bus.read("Present_Current", a.joint)
            st = bus.read("Status", a.joint)
            print("{:>5}{:>10.1f}{:>10.1f}{:>9}{:>9}{:>8}".format(
                i, goal, pos, ld, cu, st))

            if st != 0:
                reason = "Status={} 舵机报错".format(st)
                break
            if abs(ld) > a.load_limit:
                reason = "负载 {} 超过上限".format(ld)
                break
            if abs(pos - goal) > a.step * 3:
                reason = "跟不上（误差 {:.1f}deg），抬不动".format(abs(pos - goal))
                break
    except KeyboardInterrupt:
        reason = "用户中断"
    except Exception as e:
        reason = "异常: {}".format(e)
    finally:
        print("")
        print("停止原因: {}".format(reason))
        try:
            bus.disable_torque()
            print("扭矩已关闭 —— 注意臂会落下")
        except Exception as e:
            print("关扭矩失败（可能已跳闸）: {}".format(e))
        try:
            bus.disconnect(False)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
