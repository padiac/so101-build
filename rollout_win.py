#!/usr/bin/env python3
"""
Windows 上部署训练好的策略 —— 打上重试补丁后调用 lerobot 官方 rollout 入口。

和直接跑 lerobot-rollout 的唯一区别：先 apply 了 lerobot_patch，
让 enable_torque/disable_torque 带重试，避开上电涌浪导致的丢包。
（exit_early 那个补丁在这里是空转，无害。）

参数完全同 lerobot-rollout。一般通过 robot.ps1 eval 调用。

注意：Windows 这边的 torch 是 CPU 版。ACT 一次前向实测 222 ms，
但它一次输出 100 步动作块，每 3.3 秒才真正推理一次，其余 tick 从队列里取，
所以不是每帧 222 ms。表现为每 3.3 秒一次约 0.2 秒的顿挫。
装 CUDA 版 torch 可以消掉。
"""

import os
import sys

import lerobot_patch

lerobot_patch.apply()

# SHOW_BLOCK_BOX=1 overlays the yellow-block detection on the Rerun feed so the
# detector can be watched while the arm works. It is an instrument for us only --
# ACT maps pixels to actions end to end and never sees this box.
if os.environ.get("SHOW_BLOCK_BOX") == "1":
    from lerobot.rollout.strategies import core, episodic  # noqa: F401,E402  (import before patching)
    lerobot_patch.apply_block_overlay()

from lerobot.scripts.lerobot_rollout import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
