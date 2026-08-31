# attach-usb.ps1 -- Forward the SO-101 servo driver board from Windows into WSL.
#
# NOTE: ASCII-only on purpose. Windows PowerShell 5.1 decodes .ps1 files as ANSI
# (GBK on this machine) unless they carry a UTF-8 BOM, which mangles non-ASCII
# comments and breaks the parser. Keep this file ASCII.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File attach-usb.ps1
#   powershell -ExecutionPolicy Bypass -File attach-usb.ps1 -BusId 1-7
#
# Re-run after every USB unplug/replug: Windows re-enumerates and the
# forwarding is dropped.

param(
    [string]$BusId = "",
    [string]$Distro = "Ubuntu"
)

# ---- Locate usbipd ----
# A PowerShell window opened before usbipd-win was installed will not have it
# on PATH. Refresh from the registry, then fall back to the default install dir.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

$usbipd = "usbipd"
if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    $fallback = "C:\Program Files\usbipd-win\usbipd.exe"
    if (Test-Path $fallback) {
        $usbipd = $fallback
    }
    else {
        Write-Host "usbipd not found. Install it with:" -ForegroundColor Red
        Write-Host "  winget install --exact --id dorssel.usbipd-win"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# ---- Elevate if needed (usbipd bind requires admin) ----
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($id)
$adminRole = [Security.Principal.WindowsBuiltInRole]::Administrator

if (-not $principal.IsInRole($adminRole)) {
    Write-Host "Elevating (usbipd bind needs administrator)..." -ForegroundColor Yellow
    $argList = "-ExecutionPolicy Bypass -NoExit -File `"$PSCommandPath`""
    if ($BusId -ne "") { $argList = $argList + " -BusId " + $BusId }
    Start-Process powershell.exe -Verb RunAs -ArgumentList $argList
    exit
}

Write-Host ""
Write-Host "=== usbipd list ===" -ForegroundColor Cyan
& $usbipd list

# ---- Pick the device ----
$targets = @()

if ($BusId -ne "") {
    $targets = @($BusId)
}
else {
    $lines = & $usbipd list
    foreach ($line in $lines) {
        if ($line -notmatch '^\s*(\d+-\d+)\s') { continue }
        $thisBus = $Matches[1]

        # Skip things that are definitely not the servo board.
        if ($line -match 'Bluetooth|Input Device|LED Controller|Receiver|Descriptor Request Failed') {
            continue
        }
        if ($line -match 'Serial|COM\d|CH34|CH9|CP210|FT232|USB-SERIAL|ACM|USB2\.0-Ser') {
            $targets = $targets + $thisBus
        }
    }
}

if ($targets.Count -eq 0) {
    Write-Host ""
    Write-Host "No serial device auto-detected." -ForegroundColor Red
    Write-Host ""
    Write-Host "Checklist:"
    Write-Host "  1. USB cable connected to the driver board?"
    Write-Host "  2. Board powered? (12V for follower / 5V for leader -- do NOT mix)"
    Write-Host "  3. Both jumpers on channel B (USB)?"
    Write-Host "  4. CH34x / CP210x driver installed on Windows?"
    Write-Host ""
    Write-Host "If you can see the board in the list above, pass its BUSID directly:"
    Write-Host "  .\attach-usb.ps1 -BusId 1-7" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# ---- Bind + attach ----
foreach ($bus in $targets) {
    Write-Host ""
    # Clear any stale attach record first. After a physical unplug/replug the
    # host can still report "Attached" while WSL has no device at all, and a
    # fresh attach then silently does nothing. Detaching first makes this
    # script self-healing.
    Write-Host ("--> detaching " + $bus + " (clear stale state)") -ForegroundColor DarkGray
    & $usbipd detach --busid $bus 2>&1 | Out-Null
    Start-Sleep -Milliseconds 500

    Write-Host ("--> binding " + $bus) -ForegroundColor Green
    & $usbipd bind --busid $bus
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    (already bound, or bind failed -- continuing)" -ForegroundColor DarkGray
    }

    Write-Host ("--> attaching " + $bus + " to WSL") -ForegroundColor Green
    & $usbipd attach --wsl --busid $bus
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("    attach FAILED for " + $bus) -ForegroundColor Red
    }
}

# ---- Verify inside WSL ----
# Enumeration inside WSL takes a second or two after attach returns. Checking
# immediately gives a false "none found", so poll instead of asking once.
Write-Host ""
Write-Host "=== serial devices inside WSL ===" -ForegroundColor Cyan

$found = $false
for ($try = 1; $try -le 10; $try++) {
    $out = wsl -d $Distro -- bash -c "ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null"
    if ($out) {
        Write-Host ""
        wsl -d $Distro -- bash -c "ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null"
        $found = $true
        break
    }
    Write-Host -NoNewline "."
    Start-Sleep -Seconds 1
}

if (-not $found) {
    Write-Host ""
    Write-Host "none found after 10s" -ForegroundColor Red
}

Write-Host ""
Write-Host "If nothing showed up, debug with:" -ForegroundColor Yellow
Write-Host "  wsl -d Ubuntu -- lsusb"
Write-Host "  wsl -d Ubuntu -- bash -c 'dmesg | tail -20'"
Write-Host ""
Write-Host "On 'Permission denied' when opening the port:" -ForegroundColor Yellow
Write-Host "  wsl -d Ubuntu -- sudo chmod 666 /dev/ttyACM0"
Write-Host ""

Read-Host "Press Enter to exit"
