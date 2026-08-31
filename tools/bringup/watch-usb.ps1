# watch-usb.ps1 -- Watch for USB plug/unplug events on Windows.
#
# ASCII-only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File watch-usb.ps1
#   powershell -ExecutionPolicy Bypass -File watch-usb.ps1 -Seconds 60
#
# Run it, then plug/unplug the servo driver board while it counts down.
# If the numbers never move, Windows is not seeing the device at all:
# suspect the USB cable (charge-only?) or the USB port -- not the driver.

param(
    [int]$Seconds = 30
)

function Get-UsbSet {
    Get-CimInstance Win32_PnPEntity |
        Where-Object { $_.DeviceID -like 'USB\*' } |
        Select-Object -ExpandProperty DeviceID
}

$baseline = @(Get-UsbSet)
Write-Host ""
Write-Host ("Baseline: " + $baseline.Count + " USB devices.") -ForegroundColor Cyan
Write-Host ("Now plug / unplug the board. Watching for " + $Seconds + "s...") -ForegroundColor Yellow
Write-Host ""

$prev = $baseline

for ($i = 1; $i -le $Seconds; $i++) {
    Start-Sleep -Seconds 1
    $now = @(Get-UsbSet)

    $added   = @(Compare-Object $prev $now | Where-Object { $_.SideIndicator -eq '=>' })
    $removed = @(Compare-Object $prev $now | Where-Object { $_.SideIndicator -eq '<=' })

    foreach ($a in $added) {
        $dev = Get-CimInstance Win32_PnPEntity |
               Where-Object { $_.DeviceID -eq $a.InputObject }
        Write-Host ("[+] ADDED    " + $dev.Name) -ForegroundColor Green
        Write-Host ("             " + $a.InputObject) -ForegroundColor DarkGray
        if ($dev.ConfigManagerErrorCode -ne 0) {
            Write-Host ("             !! driver problem, error code " +
                        $dev.ConfigManagerErrorCode) -ForegroundColor Red
        }
    }
    foreach ($r in $removed) {
        Write-Host ("[-] REMOVED  " + $r.InputObject) -ForegroundColor Magenta
    }

    $prev = $now
    Write-Host -NoNewline "."
}

Write-Host ""
Write-Host ""

$final = @(Get-UsbSet)
$diff  = @(Compare-Object $baseline $final)

if ($diff.Count -eq 0) {
    Write-Host "RESULT: no USB change detected at all." -ForegroundColor Red
    Write-Host ""
    Write-Host "Windows never received a plug event. This is NOT a driver issue."
    Write-Host "Most likely, in order:"
    Write-Host "  1. The USB cable is charge-only (no data lines). Most common cause."
    Write-Host "     Test it: connect a phone with the same cable. If the phone only"
    Write-Host "     charges and no device appears on the PC, the cable is the problem."
    Write-Host "  2. Bad / flaky USB port. Use a rear motherboard port, not a hub"
    Write-Host "     and not a front-panel port."
    Write-Host "  3. Wrong port on the board (some boards have a separate power-only port)."
}
else {
    Write-Host "RESULT: USB activity detected." -ForegroundColor Green
    $diff | ForEach-Object {
        $tag = if ($_.SideIndicator -eq '=>') { "still present" } else { "gone" }
        Write-Host ("  [" + $tag + "] " + $_.InputObject)
    }
}

Write-Host ""
Read-Host "Press Enter to exit"
