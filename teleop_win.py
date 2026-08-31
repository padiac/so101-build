#!/usr/bin/env python3
"""
Windows 上跑遥操作 —— 打上重试补丁后调用 lerobot 官方入口。

和直接跑 lerobot-teleoperate 的唯一区别：先 apply 了 lerobot_patch，
让 enable_torque/disable_torque 带重试，避开上电涌浪导致的丢包。

用法（参数完全同 lerobot-teleoperate）：
    python teleop_win.py --robot.type=so101_follower --robot.port=COM8 ...
"""

import sys

import lerobot_patch

lerobot_patch.apply()

from lerobot.scripts.lerobot_teleoperate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
