#!/usr/bin/env bash
# 已废弃 —— 硬件相关的一切统一走 Windows 的 robot.ps1
cat <<'MSG'

  这个脚本已废弃。

  原因：WSL 走 usbipd 转发，每次拔插/重启都要重新 attach，高负载下还会断；
        而且 WSL 内核根本没有 UVC 支持，相机用不了 —— 采数据必须在 Windows。

  改用（Windows PowerShell）：

      cd E:\Repo\so101-build
      .\robot.ps1 teleop
      .\robot.ps1 health
      .\robot.ps1 scan
      .\robot.ps1 ports

  它会自动把板子从 WSL 抢回来、按序列号找 COM 口、打好重试补丁。

  WSL 从现在起只做一件事：训练（吃 GPU）。

MSG
exit 1
