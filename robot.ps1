# robot.ps1 -- Single entry point for all SO-101 hardware work, on Windows.
#
# ASCII-only (Windows PowerShell 5.1 reads .ps1 as ANSI).
#
# WHY ONE SCRIPT, AND WHY WINDOWS
#   Hardware work used to be split between WSL and Windows, and every session
#   started with "which side are the boards on?". That split was the actual
#   bug. It is now resolved permanently:
#
#     * Cameras CANNOT work in WSL at all -- the Microsoft kernel has zero
#       V4L2/UVC support (no uvcvideo, no videodev, no MEDIA config).
#       Data collection needs cameras and arms in the same process, so data
#       collection can only happen on Windows.
#     * On Windows the boards are plain COM ports. No usbipd, no vhci_hcd,
#       no stale attach state, no /dev/ttyACMn renumbering, and no dropouts
#       (measured: 300 ops, 0 failures, 0.36 ms median round trip).
#     * WSL is now used for ONE thing: training on the RTX 3080.
#
#   So: all hardware -> Windows (this script). Training -> WSL.
#
# SELF-HEALING
#   If the boards happen to be attached to WSL, this script detaches them
#   automatically. You never have to think about it again.
#
# Usage:
#   .\robot.ps1 health
#   .\robot.ps1 scan
#   .\robot.ps1 teleop                 # default 15 / 30 fps
#   .\robot.ps1 teleop -MaxRel 25 -Fps 30
#   .\robot.ps1 calibrate-follower
#   .\robot.ps1 calibrate-leader
#   .\robot.ps1 eval                   # run the trained policy (arm moves alone!)
#   .\robot.ps1 eval -Duration 60 -Policy policies\act_trim_100k
#   .\robot.ps1 eval -NoHome            # skip the homing step
#   .\robot.ps1 eval -Record            # also save what the policy saw and did
#   .\robot.ps1 ports                  # just show what is connected
#   .\robot.ps1 audit -Name so101_v3                    # dataset integrity check
#   .\robot.ps1 clips -Name so101_v3 -Episodes2 "75-81" # cut clips to review by eye
#   .\robot.ps1 drop  -Name so101_v3 -Episodes2 "77,80" # delete episodes (add -NoSwap to review first)
#   .\robot.ps1 progress                                # how far along is the WSL training run
#   .\robot.ps1 log                                     # tail the training log (-Tail N)
#   .\robot.ps1 eval -Live                              # instruction follows task.txt, changeable mid-run

param(
    [Parameter(Position = 0)]
    [ValidateSet("health", "scan", "teleop", "record", "eval", "ports",
                 "calibrate-follower", "calibrate-leader",
                 "audit", "clips", "drop", "progress", "log", "waypoint")]
    [string]$Action = "ports",

    [double]$MaxRel = 15,
    [int]$Fps = 30,

    # ---- record only ----
    [string]$Task = "Pick the yellow block and put it in the black bowl.",
    [string]$Name = "so101_pickplace",
    [int]$Episodes = 50,
    [switch]$CamCheck,
    [switch]$Resume,
    [switch]$Fresh,
    [int]$EpisodeTime = 25,
    [int]$ResetTime = 15,

    # ---- audit / clips / drop only ----
    # Named Episodes2 because -Episodes is already an int (how many to record).
    # Accepts "75-81", "3,7,12" or "all".
    [string]$Episodes2 = "all",
    [double]$ClipScale = 0.5,
    # Named Tail rather than Lines. PowerShell variable names are case
    # insensitive, so a parameter called Lines also types the local $lines in the
    # usbipd helper, and every action then died with
    # "Cannot convert System.Object[] to Int32".
    [int]$Tail = 3,
    [switch]$NoSwap,
    [switch]$Live,
    # ---- eval only ----
    # Deploy a trained policy. The follower moves on its own; the leader
    # is not connected at all. Keep a hand near the power switch.
    # Trained on the trimmed dataset (README #27). The untrimmed model plans
    # a total path of 6.2 units from the start pose -- i.e. it stands still.
    # This one plans 489.6 from the identical frame.
    # The v1 policies were trained while camera_map.json still had top and
    # wrist swapped (README #29). They are renamed *_OLDCAMMAP and must not
    # be run against the corrected map. This points at the v2 policy, which
    # does not exist until the new dataset is recorded and trained -- until
    # then eval stops with a clear message, which is the intended behaviour.
    [string]$Policy   = "policies\act_v3rand_100k",
    # Where to read the start pose from, when the checkpoint's own training copy
    # has no local twin -- so101_c6pool_304_trim is the same recordings as
    # datasets\so101_color6 with the instructions pooled, and only a person knows
    # that. Whatever is named here is still checked against the checkpoint's own
    # statistics before the arm moves, so naming the wrong one still stops.
    [string]$HomeDataset = "",
    [int]$Duration    = 30,
    [switch]$NoHome,
    # Rerun logs both camera streams every tick. In a short run that is free; in a
    # long-lived session it accumulates until it hits its 1 GiB limit and then
    # drags the whole control loop down -- measured at 6.4 Hz at the start of a
    # session and 1.9 Hz once rerun was full, against 30 fps in training.
    [switch]$NoDisplay,
    # -Record saves the rollout itself as a dataset: every frame the policy
    # actually saw and every action it actually issued. Without it a failed
    # run leaves nothing to examine and the next step is guesswork.
    # Number of policy attempts to run back to back. lerobot's 'episodic'
    # strategy already does episode + reset phases with a return to the
    # start pose between them, so this reuses the path that works rather
    # than hand-rolling a control loop.
    [int]$Trials    = 1,
    [int]$ResetTimeEval = 4,
    # Draw the block detector's box on the Rerun feed while the arm runs.
    # Purely diagnostic -- the policy never sees it.
    [switch]$ShowBox,
    [switch]$Record,
    # ACT plans chunk_size=100 actions at once. The checkpoint default runs all
    # 100 open loop -- 3.3 s at 30 fps with the cameras ignored. With only 30
    # demonstrations there is no data covering "I drifted off, how do I get
    # back", so one bad chunk puts the arm somewhere unseen and it never
    # recovers. Replanning often is the fix; it needs the GPU (CPU forward is
    # 314 ms, GPU is 18 ms).
    # Execute the whole chunk before replanning. Replanning part way
    # through re-seeds the trajectory from the arm's current pose and
    # each new plan sits slightly ahead of it, so the arm creeps to the
    # far end of the training range instead of stopping on the object.
    # A chunk on its own does not do that -- it goes out and comes back.
    # Measured: with 15, only the far position worked; with 50 (the full
    # chunk) positions 2 and 3 both work.
    [int]$ActionSteps = 50,

    # Temporal ensembling: query the policy every step and blend the overlapping
    # chunks. It removes the discontinuity at chunk boundaries. With plain
    # chunking the command restarts from the measured position at each replan,
    # and because the arm lags behind (tracking error 34.8 against 4.6 under
    # teleop) that restart yanks the command backwards -- the measured reversal
    # period matched the replan interval exactly, 0.65-0.68 s against 0.67 s.
    # Requires n_action_steps=1, so inference runs every tick: 18 ms on the GPU
    # against a 33 ms budget.
    [switch]$Ensemble,
    [double]$EnsembleCoeff = 0.01,
    [string]$Device   = "cuda",

    [string]$FollowerSerial = "5B3E090575",
    [string]$LeaderSerial   = "5B61033038"
)

# Native programs (lerobot) log to stderr. PowerShell 5.1 wraps each stderr
# line in an ErrorRecord and prints a NativeCommandError banner, which buries
# the real output. Flatten stderr to plain strings instead.
$ErrorActionPreference = "Continue"
# Run a native command, streaming its output.
#
# The exit code used to be discarded here, which is how a failed
# lerobot-edit-dataset let the caller carry on and move a good dataset out of
# the way with nothing to put back. The code is now recorded, and callers that
# must not proceed after a failure use Invoke-NativeStrict.
#
# Plain Invoke-Native still tolerates non-zero: tools/dataset/audit_episodes.py
# deliberately exits 2 when it finds problems, and that is not a script error.
function Invoke-Native {
    & $args[0] @($args[1..($args.Count - 1)]) 2>&1 | ForEach-Object { "$_" }
    $script:LastNativeExit = $LASTEXITCODE
}

function Invoke-NativeStrict {
    Invoke-Native @args
    if ($script:LastNativeExit -ne 0) {
        throw ($args[1..($args.Count - 1)] -join ' ') + " failed with exit code $script:LastNativeExit"
    }
}
Set-Location $PSScriptRoot
$venv = Join-Path $PSScriptRoot ".venv-win\Scripts"
$py = Join-Path $venv "python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Windows venv not found at $py" -ForegroundColor Red
    exit 1
}


# ---- Step 1: take the boards back from WSL if usbipd is holding them ----
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

# Prepend the venv's Scripts dir. This MUST come after the line above, which
# ASSIGNS (not appends) to $env:Path and would otherwise wipe it out.
# We invoke python.exe by absolute path instead of activating the venv, so
# console entry points there are invisible unless added explicitly. lerobot
# needs rerun.exe (the Rerun Viewer) on PATH for --display_data, otherwise:
#   RuntimeError: Failed to find Rerun Viewer executable in PATH
$env:Path = "$venv;" + $env:Path
$usbipd = "usbipd"
if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    $fb = "C:\Program Files\usbipd-win\usbipd.exe"
    if (Test-Path $fb) { $usbipd = $fb } else { $usbipd = $null }
}
if ($usbipd) {
    $lines = & $usbipd list 2>$null
    foreach ($line in $lines) {
        if ($line -match '^\s*(\d+-\d+)\s+1a86:' -and $line -match 'Attached') {
            $bus = $Matches[1]
            Write-Host "reclaiming $bus from WSL..." -ForegroundColor DarkGray
            & $usbipd detach --busid $bus 2>&1 | Out-Null
            Start-Sleep -Milliseconds 800
        }
    }
}

# ---- Step 2: resolve COM ports by SERIAL NUMBER (COM numbers move) ----
$resolve = @"
from serial.tools import list_ports
want = {'$FollowerSerial': 'F', '$LeaderSerial': 'L'}
got = {}
for p in list_ports.comports():
    if p.serial_number in want:
        got[want[p.serial_number]] = p.device
print((got.get('F') or '-') + ' ' + (got.get('L') or '-'))
"@
$pair = ((& $py -c $resolve) | Select-Object -Last 1).Trim() -split '\s+'
$FOLLOWER = $pair[0]
$LEADER = if ($pair.Count -gt 1) { $pair[1] } else { "-" }

Write-Host ""
Write-Host ("follower  {0}   (SN {1})" -f $FOLLOWER, $FollowerSerial)
Write-Host ("leader    {0}   (SN {1})" -f $LEADER, $LeaderSerial)
Write-Host ""

if ($Action -eq "ports") {
    & $py -c "from serial.tools import list_ports; [print(p.device, '|', p.description, '| SN=', p.serial_number) for p in list_ports.comports()]"
    exit 0
}

if ($FOLLOWER -eq "-" -or $LEADER -eq "-") {
    Write-Host "One or both boards not found." -ForegroundColor Red
    Write-Host "Check: USB cable seated (data cable, not charge-only), DC power on."
    Write-Host "  12V -> follower   /   5V -> leader   (never swap these)"
    Write-Host ""
    & $py -c "from serial.tools import list_ports; [print(' ', p.device, p.description, p.serial_number) for p in list_ports.comports()]"
    exit 1
}

New-Item -ItemType Directory -Force -Path logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

switch ($Action) {
    "health" {
        Invoke-Native $py health.py --leader-port $LEADER --follower-port $FOLLOWER
    }
    "scan" {
        Write-Host "--- follower ---"; Invoke-Native $py scan_motors.py --port $FOLLOWER
        Write-Host "--- leader ---";   Invoke-Native $py scan_motors.py --port $LEADER
    }
    "calibrate-follower" {
        & (Join-Path $venv "lerobot-calibrate.exe") `
            --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=my_follower
    }
    "calibrate-leader" {
        & (Join-Path $venv "lerobot-calibrate.exe") `
            --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=my_leader
    }
    "record" {
        # 1) Verify camera identity first. The two cameras have no serial
        #    numbers and are indistinguishable at the device level; if top and
        #    wrist get swapped, training still completes and loss still drops,
        #    but the policy learns the wrong mapping. Silent dataset corruption.
        # Camera check is OPT-IN (-CamCheck), not default.
        # The mapping was confirmed visually once via the contact sheet
        # (index 0 shows the gripper = wrist, index 1 shows the whole bench = top),
        # which is stronger evidence than any automated heuristic. The automated
        # check went through three broken versions (wrong criterion, wrong arm
        # moved, wrong console encoding) and cost far more time than it saved.
        # Keep it available for a periodic re-check, but never let it block recording.
        if (-not $CamCheck) {
            Write-Host "camera check skipped (add -CamCheck to run it)" -ForegroundColor DarkGray
        }
        else {
        Write-Host "=== camera check ===" -ForegroundColor Cyan
        Invoke-Native $py cams.py verify --follower-port $FOLLOWER
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "Camera check FAILED. Not recording." -ForegroundColor Red
            Write-Host "Run: .\.venv-win\Scripts\python.exe cams.py list" -ForegroundColor Yellow
            Write-Host "Or skip it: .
obot.ps1 record -SkipCamCheck ..." -ForegroundColor Yellow
            exit 1
        }
        }

        # 2) Read indices from camera_map.json rather than hardcoding them.
        # Resolve camera indices by DEVICE NAME before reading the map.
        # OpenCV indices are reassigned whenever the USB layout changes -- a
        # webcam plugged in for dictation pushed the wrist camera from index 1
        # to 2, and the stored map then pointed at the wrong device. Pinning
        # numbers only defers that; resolving by name every run fixes it for
        # good. It refuses to write a map it cannot determine confidently.
        Invoke-Native $py cams.py resolve
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Camera resolve failed -- not proceeding.' -ForegroundColor Red
            exit 1
        }

        $map = Get-Content camera_map.json -Raw | ConvertFrom-Json
        $topIdx = $map.top
        $wristIdx = $map.wrist

        # MJPG is required, not optional: both cameras share one USB 2.0 bus.
        # YUY2 at 640x480x30 needs ~18 MB/s each; two of them exceed the ~35 MB/s
        # practical ceiling of USB 2.0 and frames get dropped. MJPG compresses
        # roughly 10x, after which bandwidth is a non-issue.
        $cams = "{ top: {type: opencv, index_or_path: $topIdx, width: 640, height: 480, fps: $Fps, fourcc: MJPG, backend: 1400}, " +
                "wrist: {type: opencv, index_or_path: $wristIdx, width: 640, height: 480, fps: $Fps, fourcc: MJPG, backend: 1400} }"

        $root = Join-Path $PSScriptRoot "datasets\$Name"

        # LeRobotDataset.create uses mkdir(exist_ok=False), so a second run into
        # the same directory dies with FileExistsError. Make the two intents explicit.
        if (Test-Path $root) {
            if ($Fresh) {
                Write-Host "removing existing dataset: $root" -ForegroundColor Yellow
                Remove-Item -Recurse -Force $root
            }
            elseif (-not $Resume) {
                Write-Host ""
                Write-Host "Dataset already exists: $root" -ForegroundColor Red
                $existing = 0
                $infoPath = Join-Path $root "meta\info.json"
                if (Test-Path $infoPath) {
                    try { $existing = (Get-Content $infoPath -Raw | ConvertFrom-Json).total_episodes } catch {}
                }
                Write-Host "  it currently holds $existing episode(s)."
                Write-Host ""
                Write-Host "Pick one:" -ForegroundColor Yellow
                Write-Host "  -Resume       keep what is there and append new episodes"
                Write-Host "  -Fresh        delete it and start over"
                Write-Host "  -Name other   record into a different dataset"
                exit 1
            }
        }

        $resumeFlag = if ($Resume) { "true" } else { "false" }
        Write-Host ""
        Write-Host "task     : $Task"
        Write-Host "episodes : $Episodes  ($EpisodeTime s each, $ResetTime s reset)"
        Write-Host "dataset  : $root"
        Write-Host ""

        Invoke-Native $py record_win.py `
            --robot.type=so101_follower `
            --robot.port=$FOLLOWER `
            --robot.id=my_follower `
            --robot.max_relative_target=$MaxRel `
            --robot.cameras="$cams" `
            --teleop.type=so101_leader `
            --teleop.port=$LEADER `
            --teleop.id=my_leader `
            --dataset.repo_id="local/$Name" `
            --dataset.single_task="$Task" `
            --dataset.root="$root" `
            --dataset.num_episodes=$Episodes `
            --dataset.episode_time_s=$EpisodeTime `
            --dataset.reset_time_s=$ResetTime `
            --dataset.fps=$Fps `
            --dataset.push_to_hub=false `
            --resume=$resumeFlag `
            --display_data=true
    }
    "eval" {
        # Deploy a trained policy. NOTE: the leader arm is NOT used here -- the
        # follower is driven entirely by the policy. This is the first time the
        # arm moves with no human in the loop, so keep the first run short.

        $policyPath = $Policy
        if (-not [IO.Path]::IsPathRooted($policyPath)) {
            $policyPath = Join-Path $PSScriptRoot $Policy
        }
        if (-not (Test-Path (Join-Path $policyPath "config.json"))) {
            Write-Host ""
            Write-Host "No policy at: $policyPath" -ForegroundColor Red
            Write-Host "Expected a pretrained_model dir containing config.json." -ForegroundColor Yellow
            Write-Host "Copy one out of WSL, for example:" -ForegroundColor Yellow
            Write-Host "  wsl -d Ubuntu -- cp -r /home/padiac/lerobot-train/outputs/<run>/checkpoints/<step>/pretrained_model /mnt/e/Repo/so101-build/policies/<name>"
            exit 1
        }

        # Same camera wiring as record. The policy was trained on this exact
        # top/wrist assignment; swapping them here feeds the network mirrored
        # inputs and it fails in ways that look like bad training.
        # Resolve camera indices by DEVICE NAME before reading the map.
        # OpenCV indices are reassigned whenever the USB layout changes -- a
        # webcam plugged in for dictation pushed the wrist camera from index 1
        # to 2, and the stored map then pointed at the wrong device. Pinning
        # numbers only defers that; resolving by name every run fixes it for
        # good. It refuses to write a map it cannot determine confidently.
        Invoke-Native $py cams.py resolve
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Camera resolve failed -- not proceeding.' -ForegroundColor Red
            exit 1
        }

        $map = Get-Content camera_map.json -Raw | ConvertFrom-Json
        $cams = "{ top: {type: opencv, index_or_path: $($map.top), width: 640, height: 480, fps: $Fps, fourcc: MJPG, backend: 1400}, " +
                "wrist: {type: opencv, index_or_path: $($map.wrist), width: 640, height: 480, fps: $Fps, fourcc: MJPG, backend: 1400} }"

        Write-Host ""
        Write-Host "policy   : $policyPath"
        Write-Host "task     : $Task"
        Write-Host "duration : $Duration s   fps $Fps   max_rel $MaxRel"
        Write-Host "device   : $Device   n_action_steps $ActionSteps (replan every $([math]::Round($ActionSteps/$Fps*1000)) ms)"
        Write-Host ""
        # Computed here and echoed, rather than inlined into the command: the
        # first attempt at this switch added the parameter and never wired it to
        # the argument, so -NoDisplay ran with the rerun window still up and the
        # comparison it existed for was meaningless.
        $displayData = if ($NoDisplay) { "false" } else { "true" }
        Write-Host "display  : $displayData   (rerun window)"
        Write-Host ""
        Write-Host "THE ARM MOVES BY ITSELF. Keep a hand near the power switch." -ForegroundColor Yellow
        Write-Host ""

        # Park the arm at the pose every training episode started from.
        # Measured over the 30 episodes, that start pose is extremely tight
        # (shoulder_lift spread 0.35 units), so the policy has never seen a
        # run that began anywhere else. Starting elsewhere puts the very first
        # observation out of distribution, and ACT then plays 100 open-loop
        # steps planned from it -- the arm drifts and never reaches the object.
        # Which dataset's start pose to park at is needed whether or not we park
        # now: -Live also returns the arm here between instructions, so resolve
        # it unconditionally. Home to the pose THIS policy's training data starts
        # from, not a hardcoded default. home.py defaulted to the v1 dataset, so
        # every v2 evaluation began 117 units away in wrist_roll and outside the
        # v2 start range in shoulder_pan -- out of distribution from frame one.
        # The checkpoint records which dataset it was trained on.
        # There is no default. Falling back to some other task's dataset is the
        # bug this block exists to prevent, and a fallback is how it happened
        # twice: once from home.py's v1 default, and again on 2026-09-12, when
        # so101_color6_304_trim found no local directory and homed to so101_v2 --
        # wrist_roll +18 against the recorded -66, 84 units out and nowhere near
        # the -97..-38 the training data ever saw. The wrist camera is bolted to
        # that joint, so every rollout looked at the scene from an angle no
        # demonstration contained. Refusing to home is recoverable; homing to the
        # wrong pose looks like it worked.
        $trainCfg = Join-Path $policyPath 'train_config.json'
        $homeDs = $null
        if ($HomeDataset) {
            if (-not (Test-Path $HomeDataset)) {
                Write-Host "-HomeDataset $HomeDataset does not exist" -ForegroundColor Red
                exit 1
            }
            $homeDs = $HomeDataset
        }
        elseif (Test-Path $trainCfg) {
            $rid = (Get-Content $trainCfg -Raw | ConvertFrom-Json).dataset.repo_id
            if ($rid) {
                # $dsName, not $name: PowerShell variable names are case
                # insensitive, so assigning to $name would overwrite the -Name
                # parameter. And the separator has to be explicit -- written as
                # 'datasets' + $dsName it builds "datasetsso101_v3", Test-Path
                # always fails.
                $sep = [IO.Path]::DirectorySeparatorChar
                $dsName = ($rid -split '/')[-1] -replace '_trim$','' -replace '_nostate$',''
                # The training copy carries an episode count the local one does
                # not: so101_color6_304_trim is built from datasets\so101_color6.
                # Try the full name, then drop trailing _<digits> groups.
                $try = $dsName
                while ($true) {
                    if (Test-Path (Join-Path $PSScriptRoot ('datasets' + $sep + $try))) {
                        $homeDs = 'datasets' + $sep + $try
                        break
                    }
                    if ($try -notmatch '_\d+$') { break }
                    $try = $try -replace '_\d+$', ''
                }
                if (-not $homeDs) {
                    Write-Host "no local dataset for $rid (tried $dsName down to $try)" -ForegroundColor Red
                }
            }
        }
        if (-not $homeDs -and -not $NoHome) {
            Write-Host "cannot find the data this policy was trained on, so the" -ForegroundColor Red
            Write-Host "start pose is unknown. Copy that dataset into datasets\," -ForegroundColor Red
            Write-Host "or pass -NoHome and place the arm yourself." -ForegroundColor Red
            exit 1
        }
        # Name matching is what failed on 09-12, so do not trust it. The
        # checkpoint carries the statistics of its own training states; a start
        # pose outside that range is not this policy's data whatever the
        # directory is called.
        if ($homeDs -and -not $NoHome) {
            & $py tools\policy\pose_in_range.py $policyPath $homeDs
            if ($LASTEXITCODE -ne 0) {
                Write-Host "refusing to run: the arm would start outside this" -ForegroundColor Red
                Write-Host "policy's training range." -ForegroundColor Red
                exit 1
            }
        }

        if (-not $NoHome) {
            Write-Host "=== homing to training start pose ===" -ForegroundColor Cyan
            Write-Host "homing target from: $homeDs" -ForegroundColor DarkGray
            Invoke-Native $py home.py --port $FOLLOWER --dataset $homeDs
            if ($LASTEXITCODE -eq 1) {
                Write-Host "homing failed, not running the policy" -ForegroundColor Red
                exit 1
            }
            Write-Host ""
        }
        else {
            Write-Host "homing skipped (-NoHome)" -ForegroundColor DarkGray
        }

        # n_action_steps may not exceed the policy's chunk_size, which is baked
        # into the checkpoint at training time. Different checkpoints here use
        # different chunk sizes, so read it and clamp rather than making the
        # user remember which is which -- passing too large a value fails deep
        # inside the config parser with a stack trace that says nothing useful.
        $cfgFile = Join-Path $policyPath 'config.json'
        $chunk = (Get-Content $cfgFile -Raw | ConvertFrom-Json).chunk_size
        if ($ActionSteps -gt $chunk) {
            Write-Host "n_action_steps $ActionSteps exceeds this policy's chunk_size $chunk; using $chunk" -ForegroundColor Yellow
            $ActionSteps = [int]$chunk
        }

        $ensembleArgs = @()
        if ($Ensemble) {
            $ensembleArgs = @("--policy.temporal_ensemble_coeff=$EnsembleCoeff")
        }

        if ($ShowBox) { $env:SHOW_BLOCK_BOX = "1" } else { Remove-Item Env:SHOW_BLOCK_BOX -ErrorAction SilentlyContinue }

        # -Live points the policy at task.txt and lets the instruction change
        # mid-run: the inference engine re-reads its task on every inference, so a
        # new sentence takes effect within one chunk. See live_task.py, and write
        # to the file with say.py.
        if ($Live) {
            # Start with NO instruction. Seeding a default meant the arm began
            # working the instant it started, on whatever the default happened to
            # be -- and the default was the old put-it-in-the-bowl task. With the
            # file empty the policy simply holds position until asked for
            # something (live_task.py returns no action while it is blank).
            $taskFile = Join-Path $PSScriptRoot "task.txt"
            [IO.File]::WriteAllText($taskFile, "", (New-Object Text.UTF8Encoding($false)))
            $env:LIVE_TASK_FILE = $taskFile
            # A new instruction used to begin from wherever the previous one had
            # left the arm -- mid-reach over some other cube, a pose no episode
            # ever started from. With this set, each instruction first drives the
            # arm back to the training start pose, so every one begins in
            # distribution and has a visible start and end.
            $env:LIVE_HOME_DATASET = $homeDs
            Write-Host "live mode: idle until you send an instruction" -ForegroundColor Cyan
            Write-Host "  buttons:  .\.venv-win\Scripts\python.exe panel.py" -ForegroundColor Cyan
            Write-Host "  or type:  .\.venv-win\Scripts\python.exe say.py red" -ForegroundColor Cyan
        } else {
            Remove-Item Env:LIVE_TASK_FILE -ErrorAction SilentlyContinue
            Remove-Item Env:LIVE_HOME_DATASET -ErrorAction SilentlyContinue
        }

        $strategy = if ($Record) { "episodic" } else { "base" }
        $recordArgs = @()
        if ($Record -and $Live) {
            # The recorder stamps one instruction across the whole dataset, taken
            # from -Task. In live mode the instruction changes while it runs, so
            # the recording would carry a label that was true for none of it --
            # and anything read back from it afterwards would be scored against
            # the wrong colour.
            Write-Host "-Record and -Live cannot be combined: the recording gets" -ForegroundColor Red
            Write-Host "one instruction label, and live mode changes it mid-run." -ForegroundColor Red
            Write-Host "Record one instruction at a time with -Task instead." -ForegroundColor Red
            exit 1
        }
        if ($Record) {
            $rroot = Join-Path (Join-Path $PSScriptRoot "datasets") "rollout_probe"
            if (Test-Path $rroot) { Remove-Item -Recurse -Force $rroot }
            Write-Host "recording rollout to: $rroot" -ForegroundColor Cyan
            $recordArgs = @(
                "--dataset.repo_id=local/rollout_probe",
                "--dataset.root=$rroot",
                "--dataset.single_task=$Task",
                "--dataset.num_episodes=$Trials",
                "--dataset.episode_time_s=$Duration",
                "--dataset.reset_time_s=$ResetTimeEval",
                "--dataset.fps=$Fps",
                "--dataset.push_to_hub=false",
                # Encode while the episode runs instead of afterwards. Without
                # this each attempt ends with a multi-second pause for mp4
                # encoding, which dominates a back-to-back evaluation session.
                "--dataset.streaming_encoding=true",
                "--dataset.encoder_threads=2",
                "--strategy.smooth_leader_to_follower_handover=false",
                "--strategy.smooth_handover=false"
            )
        }

        # Camera key names are a per-checkpoint convention, not a standard: the ACT
        # policies here use top/wrist, lerobot/smolvla_base uses camera1/2/3. Derive
        # the map from the checkpoint rather than remembering which wants which.
        # --ps escapes the inner quotes: PowerShell strips bare ones on the way to a
        # native command, and the resulting {a:b} parses as an empty map -- the
        # rename silently does nothing while the startup check still passes.
        $renameArgs = @()
        $rn = & $py camera_rename.py $policyPath top wrist --ps 2>$null
        if ($rn -and $rn.Trim()) {
            $renameArgs = @("--rename_map=$($rn.Trim())")
            Write-Host "camera rename applied (policy uses camera1/camera2)" -ForegroundColor DarkGray
        }

        Invoke-Native $py rollout_win.py `
            --strategy.type=$strategy `
            --policy.path="$policyPath" `
            --device=$Device `
            --policy.n_action_steps=$(if ($Ensemble) { 1 } else { $ActionSteps }) `
            --robot.type=so101_follower `
            --robot.port=$FOLLOWER `
            --robot.id=my_follower `
            --robot.max_relative_target=$MaxRel `
            --robot.cameras="$cams" `
            --task="$Task" `
            --fps=$Fps `
            --duration=$Duration `
            --display_data=$displayData `
            --play_sounds=false `
            @renameArgs @ensembleArgs @recordArgs
    }
    "teleop" {
        $log = "logs\teleop_$stamp.log"
        "=== health before ===" | Tee-Object -FilePath $log
        Invoke-Native $py health.py --leader-port $LEADER --follower-port $FOLLOWER |
            Tee-Object -FilePath $log -Append
        "" | Tee-Object -FilePath $log -Append
        "=== teleop  maxrel=$MaxRel fps=$Fps ===" | Tee-Object -FilePath $log -Append
        # teleop_win.py applies lerobot_patch first: enable_torque/disable_torque
        # get retries, so a packet lost to servo inrush current does not abort.
        Invoke-Native $py teleop_win.py `
            --robot.type=so101_follower `
            --robot.port=$FOLLOWER `
            --robot.id=my_follower `
            --robot.max_relative_target=$MaxRel `
            --teleop.type=so101_leader `
            --teleop.port=$LEADER `
            --teleop.id=my_leader `
            --fps=$Fps | Tee-Object -FilePath $log -Append
        "" | Tee-Object -FilePath $log -Append
        "=== health after ===" | Tee-Object -FilePath $log -Append
        Invoke-Native $py health.py --leader-port $LEADER --follower-port $FOLLOWER |
            Tee-Object -FilePath $log -Append
        Write-Host ""
        Write-Host "log: $log" -ForegroundColor Cyan
    }
    "waypoint" {
        # Capture the pose the arm should pass THROUGH on its way home between
        # two instructions. Straight-line joint interpolation from wherever the
        # policy left the arm runs through the table and through the wrist
        # camera; folding up and turning the wrist first avoids that, and where
        # "up" is depends on what is on this desk, so it is captured rather than
        # guessed. Move the arm there by hand (torque is off until it moves),
        # then run this.
        Write-Host "put the arm where it should pass through on its way home," -ForegroundColor Cyan
        Write-Host "then this records that pose. It does not move the arm." -ForegroundColor Cyan
        Write-Host ""
        Invoke-NativeStrict $py home.py --port $FOLLOWER --dataset ("datasets" + [IO.Path]::DirectorySeparatorChar + $Name) --save-waypoint
    }
    "audit" {
        # Cheap integrity pass. Run it after EVERY recording session: a crash
        # leaves info.json ahead of what was written, and the failure is silent
        # -- training just cannot load the dataset.
        Invoke-Native $py tools/dataset/audit_episodes.py ("datasets/" + $Name)
    }
    "clips" {
        # v3 packs many episodes per mp4, so there is nothing to double-click.
        # This cuts one clip per episode, cameras side by side, with the episode
        # number burned in so a bad take can be identified for -Action drop.
        Invoke-Native $py tools/dataset/export_clips.py ("datasets/" + $Name) `
            --episodes $Episodes2 --scale $ClipScale
        $dir = Join-Path $PSScriptRoot ("datasets" + [IO.Path]::DirectorySeparatorChar + $Name + [IO.Path]::DirectorySeparatorChar + "clips")
        Write-Host ""
        Write-Host "clips: $dir" -ForegroundColor Cyan
        if (Test-Path $dir) { explorer $dir }
    }
    "drop" {
        # Deleting episodes is the one action here that can lose real work, so
        # every step is checked before the previous one is disturbed:
        #   1. the requested indices must exist in the dataset
        #   2. lerobot-edit-dataset must exit 0 (it never edits in place; without
        #      an explicit --new_root it writes to $HF_LEROBOT_HOME and silently
        #      leaves the source alone)
        #   3. the new dataset must exist and pass the audit
        # Only then is the old version moved aside. Skipping check 2 once emptied
        # a good dataset folder with nothing to put back.
        if ($Episodes2 -eq "all") { throw "drop needs -Episodes2, e.g. -Episodes2 ""77,80""" }
        $src = Join-Path $PSScriptRoot ("datasets" + [IO.Path]::DirectorySeparatorChar + $Name)
        $dst = $src + "_kept"
        if (-not (Test-Path (Join-Path $src "meta/info.json"))) { throw "$src is not a dataset" }
        if (Test-Path $dst) { throw "$dst already exists -- move or delete it first" }

        # "77,80" means two episodes; "77-80" means four. Expand either.
        $ids = @()
        foreach ($part in $Episodes2.Split(",")) {
            $part = $part.Trim()
            if ($part -match '^(\d+)-(\d+)$') { $ids += [int]$Matches[1]..[int]$Matches[2] }
            else { $ids += [int]$part }
        }
        $ids = $ids | Sort-Object -Unique

        # Check the indices exist before doing anything destructive. Asking to
        # delete an episode that was already deleted is the common case.
        $total = (Get-Content (Join-Path $src "meta/info.json") -Raw | ConvertFrom-Json).total_episodes
        $bad = $ids | Where-Object { $_ -lt 0 -or $_ -ge $total }
        if ($bad) {
            throw ("$Name has $total episodes, numbered 0.." + ($total - 1) +                    " -- no such episode: " + ($bad -join ", "))
        }
        $json = "[" + ($ids -join ", ") + "]"
        Write-Host ("deleting " + $ids.Count + " episode(s) " + $json + " from " + $Name) -ForegroundColor Yellow
        if ($NoSwap) { Write-Host "source will be left as-is; result goes to $dst" -ForegroundColor Cyan }
        else { Write-Host "the old version will be kept alongside as <name>_old_<timestamp>" -ForegroundColor Cyan }

        Invoke-NativeStrict $py -m lerobot.scripts.lerobot_edit_dataset `
            --repo_id ("local/" + $Name) --root $src `
            --new_repo_id ("local/" + $Name + "_kept") --new_root $dst `
            --operation.type delete_episodes --operation.episode_indices $json

        if (-not (Test-Path (Join-Path $dst "meta/info.json"))) {
            throw "$dst was not created -- source left untouched"
        }
        Write-Host ""
        Invoke-Native $py tools/dataset/audit_episodes.py ("datasets/" + $Name + "_kept")
        if ($script:LastNativeExit -ne 0) {
            throw "the new dataset did not pass the audit -- source left untouched, result kept at $dst"
        }
        $kept = (Get-Content (Join-Path $dst "meta/info.json") -Raw | ConvertFrom-Json).total_episodes
        if ($kept -ne ($total - $ids.Count)) {
            throw ("expected " + ($total - $ids.Count) + " episodes, got $kept -- source left untouched")
        }

        if ($NoSwap) {
            Write-Host ""
            Write-Host "-NoSwap: source left as-is. Result is at $dst" -ForegroundColor Yellow
            break
        }

        # Move the CONTENTS rather than renaming the folder. An Explorer window
        # or a shell sitting in the dataset directory locks that directory itself
        # but not its children, so Rename-Item fails where Move-Item succeeds.
        $old = $src + "_old_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        New-Item -ItemType Directory -Path $old -Force | Out-Null
        Get-ChildItem -Force $src | ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $old }
        Get-ChildItem -Force $dst | ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $src }
        Remove-Item -LiteralPath $dst
        Write-Host ""
        Invoke-Native $py tools/dataset/audit_episodes.py ("datasets/" + $Name)
        Write-Host ""
        Write-Host ("swapped in. previous version kept at " + $old) -ForegroundColor Green
        Write-Host "delete it once you are happy." -ForegroundColor Green
    }
    "progress" {
        # Reads the checkpoint directory in WSL, not a log file, so it works
        # from any shell and outlives whatever session launched the run.
        wsl -d Ubuntu -- bash /mnt/e/Repo/so101-build/progress.sh
    }
    "log" {
        # Read it through wsl, not \wsl$\... -- that share does not resolve
        # reliably under mirrored networking mode, and current.log is a symlink.
        wsl -d Ubuntu -- bash /mnt/e/Repo/so101-build/tailtrain.sh $Tail
    }
}
