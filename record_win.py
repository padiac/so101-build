#!/usr/bin/env python3
"""
Windows 上采数据 —— 打上重试补丁后调用 lerobot 官方 record 入口。

和直接跑 lerobot-record 的唯一区别：先 apply 了 lerobot_patch，
让 enable_torque/disable_torque 带重试，避开上电涌浪导致的丢包。

参数完全同 lerobot-record。一般通过 robot.ps1 record 调用。
"""

import sys

import lerobot_patch

lerobot_patch.apply()

from lerobot.scripts.lerobot_record import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
