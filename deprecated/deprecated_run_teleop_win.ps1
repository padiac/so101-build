# run_teleop_win.ps1 -- Teleoperation on Windows, native COM ports, no usbipd.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI).
#
# Why Windows instead of WSL:
#   WSL needs usbipd to forward the serial adapters, and that link has proven
#   unstable under sustained 30 Hz polling -- the leader board kept dropping
#   with "vhci_hcd: urb->status -104" and the device node vanishing.
#   On Windows the boards are plain COM ports. No forwarding layer, no failure.
#   Cameras also only work here (the WSL kernel has no UVC support at all).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_teleop_win.ps1
#   powershell -ExecutionPolicy Bypass -File run_teleop_win.ps1 -MaxRel 25 -Fps 30
#
# Port mapping is by SERIAL NUMBER, not COM index -- COM numbers can be
# reassigned when you replug. The script resolves them every run.

param(
    [double]$MaxRel = 15,
    [int]$Fps = 30,
    [string]$FollowerSerial = "5B3E090575",
    [string]$LeaderSerial   = "5B61033038"
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$venv = Join-Path $PSScriptRoot ".venv-win\Scripts"
$py = Join-Path $venv "python.exe"

# ---- Resolve COM ports by serial number ----
$resolve = @"
import sys
from serial.tools import list_ports
want = {'$FollowerSerial': 'FOLLOWER', '$LeaderSerial': 'LEADER'}
found = {}
for p in list_ports.comports():
    if p.serial_number in want:
        found[want[p.serial_number]] = p.device
print(found.get('FOLLOWER', '') + ' ' + found.get('LEADER', ''))
"@
$pair = (& $py -c $resolve).Trim() -split '\s+'
$follower = $pair[0]
$leader = if ($pair.Count -gt 1) { $pair[1] } else { "" }

if (-not $follower -or -not $leader) {
    Write-Host "Could not resolve both boards by serial number." -ForegroundColor Red
    Write-Host "  follower ($FollowerSerial): $follower"
    Write-Host "  leader   ($LeaderSerial): $leader"
    Write-Host ""
    Write-Host "If the boards are attached to WSL, detach them first:" -ForegroundColor Yellow
    Write-Host "  usbipd detach --busid <BUSID>"
    Write-Host ""
    & $py -c "from serial.tools import list_ports; [print(p.device, p.description, p.serial_number) for p in list_ports.comports()]"
    Read-Host "Press Enter to exit"
    exit 1
}

New-Item -ItemType Directory -Force -Path logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs\teleop_win_$stamp.log"

$header = @"
==========================================================
run_teleop_win.ps1  $(Get-Date)
follower = $follower  (SN $FollowerSerial)
leader   = $leader  (SN $LeaderSerial)
max_relative_target = $MaxRel   fps = $Fps
==========================================================
"@
$header | Tee-Object -FilePath $log

"########## HEALTH BEFORE ##########" | Tee-Object -FilePath $log -Append
& $py health.py --leader-port $leader --follower-port $follower 2>&1 |
    Tee-Object -FilePath $log -Append

"" | Tee-Object -FilePath $log -Append
"########## TELEOP START ##########" | Tee-Object -FilePath $log -Append

# Use teleop_win.py, not lerobot-teleoperate.exe: it applies lerobot_patch
# first, which gives enable_torque/disable_torque a retry count. Without that,
# a single packet lost to the inrush current when servos energise aborts the
# whole connect (observed: "Failed to write 'Lock' on id_=5 ... after 1 tries").
& $py teleop_win.py `
    --robot.type=so101_follower `
    --robot.port=$follower `
    --robot.id=my_follower `
    --robot.max_relative_target=$MaxRel `
    --teleop.type=so101_leader `
    --teleop.port=$leader `
    --teleop.id=my_leader `
    --fps=$Fps 2>&1 | Tee-Object -FilePath $log -Append

"" | Tee-Object -FilePath $log -Append
"########## HEALTH AFTER ##########" | Tee-Object -FilePath $log -Append
& $py health.py --leader-port $leader --follower-port $follower 2>&1 |
    Tee-Object -FilePath $log -Append

Write-Host ""
Write-Host "log saved: $log" -ForegroundColor Cyan
