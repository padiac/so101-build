#!/usr/bin/env python3
"""
扫描总线上的舵机 —— 装配前的验收检查。

把一条臂的 6 个舵机菊花链全部接上，跑这个脚本，确认 ID 1..6 都在、
没有重号、没有漏号。**在装进结构件之前查**，事后再发现就得拆机。

用法：
    python scan_motors.py                    # 默认 /dev/ttyACM0
    python scan_motors.py --port /dev/ttyACM1

接线：驱动板 -> shoulder_pan(1) -> shoulder_lift(2) -> elbow_flex(3)
              -> wrist_flex(4) -> wrist_roll(5) -> gripper(6)
"""

import argparse
import sys

EXPECTED = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    args = parser.parse_args()

    from lerobot.motors.feetech.feetech import FeetechMotorsBus

    print(f"\n扫描 {args.port} ...\n")
    try:
        found = FeetechMotorsBus.scan_port(args.port)
    except Exception as e:
        print(f"扫描失败: {type(e).__name__}: {e}", file=sys.stderr)
        print("\n检查：板子通电了吗 / USB 转发还在吗 (重跑 attach-usb.ps1) /", file=sys.stderr)
        print("      端口对不对 (ls /dev/ttyACM*)", file=sys.stderr)
        return 1

    # scan_port 返回 {baudrate: [id, ...]}
    ids = sorted({i for id_list in found.values() for i in id_list})

    for baudrate, id_list in sorted(found.items()):
        if id_list:
            print(f"  波特率 {baudrate}: 发现 ID {sorted(id_list)}")

    if not ids:
        print("  没有发现任何舵机。")
        print("\n检查：菊花链接好了吗 / 第一根线接到驱动板了吗 / DC 电源")
        return 1

    print("\n" + "=" * 46)
    ok = True
    for expected_id, name in EXPECTED.items():
        mark = "OK " if expected_id in ids else "缺失"
        if expected_id not in ids:
            ok = False
        print(f"  ID {expected_id}  {name:15s}  {mark}")

    extra = [i for i in ids if i not in EXPECTED]
    if extra:
        ok = False
        print(f"\n  意外的 ID: {extra}  <- 有舵机没设过 ID（新舵机默认 ID=1）")

    if len(found) > 1:
        ok = False
        print(f"\n  警告: 在多个波特率上发现设备 {sorted(found)}")
        print("        说明有舵机的波特率没被统一，重跑 setup_motors.py")

    print("=" * 46)
    if ok:
        print("\n全部正确，可以开始装配。\n")
        return 0

    print("\n有问题。把缺失/异常的舵机单独接上重设：")
    print("  python setup_motors.py follower <motor_name>\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
