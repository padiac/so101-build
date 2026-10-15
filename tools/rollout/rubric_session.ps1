# Run a planned batch of recorded evaluation trials, one trial per invocation.
#
# Why one trial per robot.ps1 call: -Record -Trials N keeps the arm wherever the
# previous episode left it during the reset gap, so trial 2 onward would start
# out of distribution. One call per trial homes every time.
#
# Why the copy: -Record deletes datasets\rollout_probe before each run, so each
# trial is moved to datasets\rollouts\<session>\<NN>_<label> straight after.
#
# Safety order: speak, wait for the person to reset the cubes, speak "hands off",
# THEN home and run. Homing moves the arm; nobody should be reaching into the
# bowl when it does.
#
# ASCII only: PowerShell 5.1 reads .ps1 files as the system codepage. Chinese
# spoken cues are built from code points at run time for the same reason.
#
#   .\tools\rollout\rubric_session.ps1 -Session s01 -Only 1      # one trial, to check the pipeline
#   .\tools\rollout\rubric_session.ps1 -Session s01 -From 2      # the rest
#
# Comparing another colour checkpoint under the same conditions -- same trials per
# colour, duration, reset time, seed and home pose; only the policy differs:
#   .\tools\rollout\rubric_session.ps1 -Session s02 -Arms colour -ColourPolicy policies\smolvla_color6_304_100k
param(
    [string]$Session = ("s" + (Get-Date -Format "yyyyMMdd_HHmm")),
    [int]$From = 1,
    [int]$Only = 0,
    [int]$ResetSeconds = 15,
    [int]$Duration = 30,
    [int]$Seed = 13,
    [string]$ColourPolicy = "policies\smolvla_color6_453_40k",
    [string]$PooledPolicy = "policies\smolvla_c6pool_453_40k",
    # Which arms of the plan to run: "colour,pooled" as in s01, or just one.
    # Not called -Models: PowerShell names are case insensitive, so it would be
    # the same variable as $models below and the table would be cast to a string.
    [string]$Arms = "colour,pooled"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = ".\.venv-win\Scripts\python.exe"

function U([int[]]$cp) { -join ($cp | ForEach-Object { [char]$_ }) }
$ZH = @{
    trial   = U 0x7B2C          # di
    ci      = U 0x6B21          # ci
    colour  = U 0x989C,0x8272,0x7248   # yan se ban
    pooled  = U 0x5408,0x5E76,0x7248   # he bing ban
    red     = U 0x7EA2,0x8272
    blue    = U 0x84DD,0x8272
    any     = U 0x4EFB,0x610F,0x65B9,0x5757
    reset   = U 0x8BF7,0x6309,0x53C2,0x8003,0x7167,0x6446,0x597D,0x65B9,0x5757   # qing an can kao zhao bai hao fang kuai
    after   = U 0x79D2,0x540E,0x5F00,0x59CB  # miao hou kai shi
    go      = U 0x5F00,0x59CB,0xFF0C,0x8BF7,0x628A,0x624B,0x62FF,0x5F00   # kai shi, qing ba shou na kai
    done    = U 0x5168,0x90E8,0x5B8C,0x6210   # quan bu wan cheng
}

Add-Type -AssemblyName System.Speech
$tts = New-Object System.Speech.Synthesis.SpeechSynthesizer
$zhVoice = $tts.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like "zh*" } | Select-Object -First 1
if ($zhVoice) { $tts.SelectVoice($zhVoice.VoiceInfo.Name) }
function Say([string]$text) { try { $tts.Speak($text) } catch { [console]::Beep(880, 400) } }

$COLOUR_TASK = "Pick the {0} cube out of the bowl and put it on the table."
$POOL_TASK   = "Pick a cube out of the bowl and put it on the table."
$models = @{
    colour = $ColourPolicy
    pooled = $PooledPolicy
}
$armList = @($Arms -split "," | ForEach-Object { $_.Trim() })
foreach ($a in $armList) {
    if (-not $models.ContainsKey($a)) { Write-Host "unknown model arm '$a'" -ForegroundColor Red; exit 2 }
    if (-not (Test-Path (Join-Path $models[$a] "config.json"))) {
        Write-Host "no checkpoint at $($models[$a])" -ForegroundColor Red; exit 2
    }
}

# 20 trials: colour model x {red, blue} x 5, pooled model x 10. Order shuffled
# with a fixed seed so a slow drift in the room (light, battery, cube wear) does
# not line up with one condition.
$plan = @()
if ($armList -contains "colour") {
    foreach ($c in "red", "blue") { 1..5 | ForEach-Object { $plan += [pscustomobject]@{ model = "colour"; colour = $c } } }
}
if ($armList -contains "pooled") {
    1..10 | ForEach-Object { $plan += [pscustomobject]@{ model = "pooled"; colour = "any" } }
}
$rng = New-Object System.Random $Seed
$plan = $plan | Sort-Object { $rng.Next() }

$outRoot = "datasets\rollouts\$Session"
New-Item -ItemType Directory -Force $outRoot | Out-Null
$index = Join-Path $outRoot "trials.csv"
if (Test-Path $index) {
    # A session is one set of checkpoints. Resuming with -From is fine; putting a
    # different checkpoint under the same label would make the report average two
    # models and call it one.
    foreach ($pp in (Import-Csv $index | Select-Object -ExpandProperty policy -Unique)) {
        if ($models.Values -notcontains $pp) {
            Write-Host "session $Session already holds trials of $pp; use a new -Session" -ForegroundColor Red
            exit 2
        }
    }
}
if (-not (Test-Path $index)) { "n,model,colour,policy,task,dir,started,exit" | Set-Content -Encoding ascii $index }

for ($i = 0; $i -lt $plan.Count; $i++) {
    $n = $i + 1
    if ($Only -gt 0 -and $n -ne $Only) { continue }
    if ($Only -eq 0 -and $n -lt $From) { continue }
    $t = $plan[$i]
    $policy = $models[$t.model]
    $task = if ($t.model -eq "colour") { $COLOUR_TASK -f $t.colour } else { $POOL_TASK }
    $label = "{0:D2}_{1}_{2}" -f $n, $t.model, $t.colour

    Write-Host ""
    Write-Host "=== trial $n / $($plan.Count): $($t.model) $($t.colour) ===" -ForegroundColor Cyan
    Write-Host "    $task"
    $what = if ($t.model -eq "colour") { "$($ZH.colour) $($ZH[$t.colour])" } else { "$($ZH.pooled) $($ZH.any)" }
    Say "$($ZH.trial) $n $($ZH.ci), $what. $($ZH.reset), $ResetSeconds $($ZH.after)."
    Start-Sleep -Seconds $ResetSeconds
    Say $ZH.go
    Start-Sleep -Seconds 2

    $started = Get-Date -Format "HH:mm:ss"
    # Both models were trained on the same so101_color6 episodes (the pooled copy
    # only rewrites the instruction), so both home to its start pose. robot.ps1
    # cannot infer that for the pooled checkpoint and correctly refuses to guess.
    & .\robot.ps1 eval -Policy $policy -Task $task -Record -Trials 1 -Duration $Duration -NoDisplay -HomeDataset datasets\so101_color6
    $rc = $LASTEXITCODE

    $dst = Join-Path $outRoot $label
    if (Test-Path "datasets\rollout_probe\meta\info.json") {
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        Copy-Item -Recurse "datasets\rollout_probe" $dst
    } else {
        Write-Host "no recording produced for trial $n" -ForegroundColor Red
    }
    "$n,$($t.model),$($t.colour),$policy,""$task"",$dst,$started,$rc" | Add-Content -Encoding ascii $index
    if ($rc -ne 0) {
        Write-Host "robot.ps1 exited $rc on trial $n -- stopping the session" -ForegroundColor Red
        exit $rc
    }
}
Say $ZH.done
Write-Host "session $Session done -> $outRoot"
