#!/usr/bin/env python3
"""
带重试的舵机 ID 设置脚本 —— 官方 lerobot-setup-motors 的替代品。

和官方的三点区别：
  1. 默认按 ID 正序 1→2→3→4→5→6 提问（官方是倒着的 6→5→...→1）
     这样你只要照着贴好的标签顺着拿，不用每次去核对减速比。
  2. 接不上不会整个 traceback 退出，原地重试，可跳过、可中途退出。
  3. 可以只补做指定的几颗。

已设好的 ID 写在舵机 EEPROM 里，重复写同一个值无害（幂等），随时重跑都安全。

用法：
    python setup_motors.py leader                    # 正序 1..6
    python setup_motors.py follower                  # 同上
    python setup_motors.py leader --reverse          # 官方顺序 6..1
    python setup_motors.py follower wrist_flex       # 只补做某几颗
    python setup_motors.py leader --port /dev/ttyACM1

[!] 主臂是 7.4V 舵机，接 5V 电源。别拿 12V 那个。
[!] 每次只能有一颗舵机连在总线上（出厂 ID 全是 1，多接就无法区分）。
"""

import argparse
import sys

DEFAULT_PORT = "/dev/ttyACM0"

# 主臂减速比对照：ID -> (料号, 减速比)。从臂 6 颗全是 C001 / 1:345。
LEADER_GEARS = {
    1: ("C044", "1/191"),
    2: ("C001", "1/345"),
    3: ("C044", "1/191"),
    4: ("C046", "1/147"),
    5: ("C046", "1/147"),
    6: ("C046", "1/147"),
}


def build_device(arm: str, port: str):
    if arm == "follower":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        return SO101Follower(SO101FollowerConfig(port=port, id="my_follower"))
    if arm == "leader":
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        return SO101Leader(SO101LeaderConfig(port=port, id="my_leader"))
    raise ValueError(f"unknown arm: {arm}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Set Feetech servo IDs, in ID order, with retry support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("arm", choices=["follower", "leader"])
    ap.add_argument("motors", nargs="*", help="只做这几颗；留空 = 全部")
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="用官方顺序（gripper 先，shoulder_pan 最后）",
    )
    args = ap.parse_args()

    device = build_device(args.arm, args.port)
    bus = device.bus

    # 按 ID 升序 = 1,2,3,4,5,6（默认）。--reverse 用官方倒序。
    ordered = sorted(bus.motors, key=lambda m: bus.motors[m].id)
    if args.reverse:
        ordered = list(reversed(ordered))

    if args.motors:
        unknown = [m for m in args.motors if m not in bus.motors]
        if unknown:
            print(f"未知的电机名: {unknown}", file=sys.stderr)
            print(f"可选: {ordered}", file=sys.stderr)
            return 1
        targets = args.motors
    else:
        targets = ordered

    print(f"\n端口 {args.port}   臂 {args.arm}   "
          f"顺序 {'官方倒序' if args.reverse else 'ID 正序 1→6'}")
    print(f"待设置 {len(targets)} 颗\n")
    print("提醒：一次只接一颗舵机，且它不要再串接别的。")
    print("      每次回车前扫一眼 DC 电源线有没有松。\n")

    done, skipped = [], []

    for i, motor in enumerate(targets, 1):
        target_id = bus.motors[motor].id
        gear = ""
        if args.arm == "leader" and target_id in LEADER_GEARS:
            part, ratio = LEADER_GEARS[target_id]
            gear = f"  [{part} · {ratio}]"

        while True:
            prompt = (
                f"[{i}/{len(targets)}] 接上标签「{target_id}」的舵机"
                f"{gear}  →  {motor}\n"
                f"          回车确认 [s=跳过, q=退出]: "
            )
            try:
                ans = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n中断。")
                _summary(done, skipped, targets, args.arm)
                return 130

            if ans == "q":
                _summary(done, skipped, targets, args.arm)
                return 0
            if ans == "s":
                skipped.append(motor)
                print(f"  跳过 {motor}\n")
                break

            try:
                bus.setup_motor(motor)
            except Exception as e:
                print(f"  x 失败: {type(e).__name__}: {e}")
                print("    检查：3-pin 线两头插到底了吗 / DC 电源松了吗 /")
                print("          是不是不小心串了第二颗舵机。回车重试。\n")
                continue

            print(f"  v {motor} -> ID {bus.motors[motor].id}\n")
            done.append(motor)
            break

    _summary(done, skipped, targets, args.arm)
    return 0


def _summary(done, skipped, targets, arm) -> None:
    print("\n" + "=" * 54)
    print(f"成功 {len(done)}: {', '.join(done) if done else '(无)'}")
    if skipped:
        print(f"跳过 {len(skipped)}: {', '.join(skipped)}")
    remaining = [m for m in targets if m not in done and m not in skipped]
    if remaining:
        print(f"未处理 {len(remaining)}: {', '.join(remaining)}")
        print("\n补做命令：")
        print(f"  python setup_motors.py {arm} {' '.join(remaining)}")
    else:
        print("\n全部完成。串成菊花链后跑验收：")
        print("  python scan_motors.py")
    print("=" * 54 + "\n")


if __name__ == "__main__":
    sys.exit(main())
