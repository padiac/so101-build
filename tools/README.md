# tools

Diagnostics. Nothing here is needed to run the robot -- `robot.ps1` at the repo
root does that. These exist to answer specific questions when something is wrong,
and most were written in response to a concrete failure. The comment at the top
of each says which one.

**Run them from the repo root**, not from this directory: they resolve
`datasets/...` and `policies/...` relative to the working directory.

```bash
cd E:\Repo\so101-build
.\.venv-win\Scripts\python.exe tools\rollout\analyze_rollout.py
```

Checkpoint and dataset defaults point at the current ones (`act_v3_100k`,
`so101_v3`); most take `--ckpt` / `--root` to look at an older pair.

## rollout/ -- after a run on the robot

Record one first with `robot.ps1 eval -Record`, which writes `datasets/rollout_probe`.

| script | answers |
|---|---|
| `analyze_rollout.py` | Was the observation in distribution? Do online actions match offline prediction? What is the oscillation period? |
| `tracking_error.py` | Commanded vs measured position -- did the arm execute what it was told? |
| `motion_profile.py` | Command speed against the demonstrations, and how often the trajectory reverses |
| `exec_vs_pred.py` | Replays the recorded observations through the policy: did anything between the model and the servos alter the action? |

## policy/ -- what the model has learned, offline

No robot needed.

| script | answers |
|---|---|
| `eval_offline.py` | Prediction error against the recorded actions, per joint |
| `per_group.py` | Does it handle each recorded block position, or only some? |
| `uses_vision.py` | Swap the images, keep the state: does the plan change at all? |
| `bias_check.py` | Is the error a pull toward the average, or random? |
| `compare_policies.py` | Two checkpoints planning from the same observation |
| `can_it_grasp.py` | Shown the moment before a grasp, does it plan the descent and the close? |
| `howclose.py` | How near did a rollout get to a demonstrated pre-grasp pose? |
| `chunk_profile.py` | How much of a chunk's motion happens in its opening steps? |
| `plan_agreement.py` | Do successive replans agree on a direction? |
| `freeze_probe.py` | When the arm stopped, was it still planning anything? |
| `live_probe.py` / `probe_live.py` | Feed a live or recorded frame and read out the plan |
| `stage_feasible.py` | Could the reach be computed from the block's pixel position instead of learned? |

## dataset/ -- before training

| script | answers |
|---|---|
| `audit_episodes.py` | Integrity: orphaned staging dirs, gaps in `episode_index`, meta/data/video disagreement, short episodes |
| `renumber_episodes.py` | Closes gaps in `episode_index` and resyncs `info.json` after a crash or re-record |
| `export_clips.py` | Cuts one reviewable mp4 per episode, cameras side by side, episode number burned in |
| `check_take.py` | Dead time at the head, episode-length consistency, leader arm in frame |
| `onset.py` | How long each episode sits still before motion starts |
| `verify_detector.py` | Draws the block detector's box on every episode's opening frame |
| `grasp_landmark.py` | Locates the grasp by watching the block go still in the wrist view |
| `label_consistency.py` | Does the block's pixel position predict the reach angle? |
| `deploy_check.py` | Per-step command deltas vs `max_relative_target`; spread of start poses |
| `domain_gap.py` | Live camera appearance against the training frames |
| `layout_ref.py` / `wrist_compare.py` / `grasp_view.py` | Contact sheets for judging by eye |

## camera/

| script | answers |
|---|---|
| `cam_matrix.py` | Every index, alone and combined, on each backend -- for when a feed goes black |
| `check_cam_match.py` | Compares live frames against the training footage to catch a swapped mapping |

## bringup/ -- hardware, mostly one-time

Servo ID setup, health checks, torque and range diagnostics, USB watching,
lens focusing. `setup_motors.py` must run **before** mechanical assembly: IDs
live in servo EEPROM and the motors are unreachable once built in.
